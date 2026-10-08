import asyncio
import inspect
import json
import logging
import re
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7
from starlette.concurrency import run_in_threadpool

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from core import settings
from evidencegraph.access import PaperAccessStore
from evidencegraph.conversations import ConversationStore, merge_thread_summaries
from evidencegraph.papers import PaperStore
from evidencegraph.retrieval import is_contextual_followup, retrieve_paper_blocks
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    EvidenceContextInput,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)
from service.agui import router as agui_router
from service.auth import get_optional_account
from service.auth import router as auth_router
from service.conversations import get_conversation_store, record_chat_workspace
from service.conversations import router as conversations_router
from service.identity import assert_session_owner, get_chat_identity, strict_identity_enabled
from service.papers import get_guest_identity, get_paper_access_store, get_paper_store
from service.papers import router as papers_router
from service.threads import list_user_threads
from service.utils import (
    convert_message_content_to_string,
    ensure_model_available,
    langchain_to_chat_message,
    messages_from_checkpoint,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)
_active_chat_tasks: dict[str, set[asyncio.Task[Any]]] = {}
_deleting_chat_threads: set[str] = set()


def _track_chat_task(thread_id: str | None) -> None:
    if thread_id is None:
        return
    if thread_id in _deleting_chat_threads:
        raise HTTPException(status_code=409, detail="Conversation deletion is in progress")
    task = asyncio.current_task()
    if task is not None:
        _active_chat_tasks.setdefault(thread_id, set()).add(task)


def _untrack_chat_task(thread_id: str | None) -> None:
    if thread_id is None:
        return
    task = asyncio.current_task()
    if task is not None:
        tasks = _active_chat_tasks.get(thread_id)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                _active_chat_tasks.pop(thread_id, None)


async def _stop_chat_tasks(thread_id: str) -> None:
    if thread_id in _deleting_chat_threads:
        raise HTTPException(status_code=409, detail="Conversation deletion is already in progress")
    _deleting_chat_threads.add(thread_id)
    current = asyncio.current_task()
    tasks = tuple(task for task in _active_chat_tasks.get(thread_id, ()) if task is not current)
    for task in tasks:
        task.cancel()
    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=409, detail="Chat generation has not stopped; retry deletion shortly"
            ) from exc


def _checkpoint_document_ids(checkpoint: Any) -> set[str]:
    """Find explicit PDF references in saved chat turns, including legacy threads."""
    if checkpoint is None:
        return set()
    document_ids: set[str] = set()
    for message in messages_from_checkpoint(checkpoint.checkpoint):
        extra = message.additional_kwargs
        context = extra.get("evidence_context")
        candidates = [extra.get("document_id")]
        if isinstance(context, dict):
            candidates.append(context.get("document_id"))
        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, str) or re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
                raise HTTPException(status_code=409, detail="Saved paper reference is invalid")
            document_ids.add(candidate)
    return document_ids


def _require_chat_paper_access(
    user_input: UserInput,
    guest_id: str,
    access: PaperAccessStore,
) -> None:
    document_id = user_input.document_id
    if user_input.evidence_context is not None:
        document_id = user_input.evidence_context.document_id
    if document_id is not None and not access.can_read(guest_id, document_id):
        raise HTTPException(status_code=404, detail="Paper not found")


async def _require_chat_thread_owner(
    user_input: UserInput,
    agent_id: str,
    conversations: ConversationStore,
    chat_identity: str | None,
) -> None:
    assert_session_owner(user_input.user_id, chat_identity)
    if not strict_identity_enabled() or user_input.thread_id is None:
        return
    workspace = await run_in_threadpool(conversations.get, user_input.thread_id)
    if workspace is not None and workspace.user_id != chat_identity:
        raise HTTPException(status_code=404, detail="Conversation not found")
    checkpointer = getattr(get_agent(agent_id), "checkpointer", None)
    if checkpointer:
        checkpoint = await checkpointer.aget_tuple(
            RunnableConfig(configurable={"thread_id": user_input.thread_id})
        )
        if checkpoint and checkpoint.metadata.get("user_id") != chat_identity:
            raise HTTPException(status_code=404, detail="Conversation not found")


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            if not settings.AUTH_SECRET:
                logger.warning(
                    "AUTH_SECRET is not configured — all API endpoints are unauthenticated. "
                    "Set AUTH_SECRET in your environment to enable bearer token authentication."
                )

            # Configure agents with both memory components and async loading
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # Continue with other agents rather than failing startup

                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(dependencies=[Depends(verify_bearer)])
# AG-UI protocol endpoints inherit the same bearer auth - see service/agui.py
router.include_router(agui_router)
router.include_router(auth_router)
router.include_router(papers_router)
router.include_router(conversations_router)


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


@router.get("/session")
def session(
    guest_id: Annotated[str, Depends(get_guest_identity)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
) -> dict[str, str | bool]:
    """Tell the browser its server-issued identity without exposing the bearer cookie."""
    return {
        "user_id": (account_id or guest_id) if settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT else guest_id,
        "identity_enforced": strict_identity_enabled(),
        "chat_requires_login": settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT,
        "authenticated": account_id is not None,
    }


async def _handle_input(
    user_input: UserInput,
    agent: AgentGraph,
    agent_id: str,
    evidence_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        ensure_model_available(user_input.model)
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        metadata={"user_id": user_id, "agent_id": agent_id},
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)

    if (
        evidence_context is not None
        and evidence_context.get("source_mode") == "automatic"
        and is_contextual_followup(user_input.message)
    ):
        previous_messages = state.values.get("messages", [])
        for previous_message in reversed(previous_messages):
            if not isinstance(previous_message, HumanMessage):
                continue
            previous_context = previous_message.additional_kwargs.get("evidence_context")
            if (
                isinstance(previous_context, dict)
                and previous_context.get("document_id") == evidence_context["document_id"]
                and previous_context.get("source_mode", "selected") == "selected"
            ):
                evidence_context = previous_context
            break

    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        additional_kwargs: dict[str, Any] = (
            {"evidence_context": evidence_context} if evidence_context is not None else {}
        )
        if evidence_context is not None:
            additional_kwargs["document_id"] = evidence_context["document_id"]
        elif user_input.document_id is not None:
            additional_kwargs["document_id"] = user_input.document_id
        input = {
            "messages": [
                HumanMessage(
                    content=user_input.message,
                    additional_kwargs=additional_kwargs,
                )
            ]
        }

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


def _resolve_evidence_context(
    selection: EvidenceContextInput | None,
    store: PaperStore,
    document_id: str | None = None,
    query: str = "",
) -> dict[str, Any] | None:
    """Resolve selected IDs or retrieve bounded passages from the current paper."""
    current_document_id = selection.document_id if selection is not None else document_id
    if current_document_id is None:
        return None
    if selection is not None and document_id is not None and selection.document_id != document_id:
        raise HTTPException(status_code=422, detail="Selected evidence belongs to another paper")
    try:
        record = store.get(current_document_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        if selection is None:
            parsed = store.get_parsed_document(current_document_id)
            resolved_blocks = retrieve_paper_blocks(parsed, query)
            if not resolved_blocks:
                raise HTTPException(status_code=422, detail="No paper passages fit the evidence budget")
        else:
            selected_blocks = []
            for block_id in selection.block_ids:
                block = store.get_block(current_document_id, block_id)
                if block is None:
                    raise HTTPException(status_code=404, detail="Evidence block not found")
                selected_blocks.append(block)
            resolved_blocks = tuple(selected_blocks)
        blocks = [
            {
                "block_id": block.block_id,
                "pages": sorted({location.page_number for location in block.locations}),
                "text": block.text,
            }
            for block in resolved_blocks
        ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Paper source is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "document_id": current_document_id,
        "source_mode": "selected" if selection is not None else "automatic",
        "blocks": blocks,
    }


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(
    user_input: UserInput,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    conversations: Annotated[ConversationStore, Depends(get_conversation_store)],
    guest_id: Annotated[str, Depends(get_guest_identity)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    agent_id: str = DEFAULT_AGENT,
) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    await _require_chat_thread_owner(user_input, agent_id, conversations, chat_identity)
    _require_chat_paper_access(user_input, guest_id, access)
    agent: AgentGraph = get_agent(agent_id)
    evidence_context = await run_in_threadpool(
        _resolve_evidence_context,
        user_input.evidence_context,
        store,
        user_input.document_id,
        user_input.message,
    )
    await record_chat_workspace(user_input, agent_id, conversations, store)
    _track_chat_task(user_input.thread_id)

    try:
        kwargs, run_id = await _handle_input(
            user_input, agent, agent_id, evidence_context
        )
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        # A run that stops on an interrupt reports it on the final event of either stream
        # mode, so check for the interrupt before falling back to the last message.
        if "__interrupt__" in response:
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        elif response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")
    finally:
        _untrack_chat_task(user_input.thread_id)


async def message_generator(
    user_input: StreamInput,
    agent_id: str = DEFAULT_AGENT,
    evidence_context: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    _track_chat_task(user_input.thread_id)

    try:
        kwargs, run_id = await _handle_input(
            user_input, agent, agent_id, evidence_context
        )
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(  # type: ignore[no-matching-overload]
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages: list[Any] = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        _untrack_chat_task(user_input.thread_id)
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(
    user_input: StreamInput,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    conversations: Annotated[ConversationStore, Depends(get_conversation_store)],
    guest_id: Annotated[str, Depends(get_guest_identity)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    agent_id: str = DEFAULT_AGENT,
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    await _require_chat_thread_owner(user_input, agent_id, conversations, chat_identity)
    _require_chat_paper_access(user_input, guest_id, access)
    evidence_context = await run_in_threadpool(
        _resolve_evidence_context,
        user_input.evidence_context,
        store,
        user_input.document_id,
        user_input.message,
    )
    await record_chat_workspace(user_input, agent_id, conversations, store)
    return StreamingResponse(
        message_generator(user_input, agent_id, evidence_context),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    if strict_identity_enabled():
        raise HTTPException(status_code=403, detail="Feedback is unavailable in strict session mode")
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/{agent_id}/history", operation_id="history_with_agent_id")
@router.post("/history")
async def history(
    input: ChatHistoryInput,
    conversations: Annotated[ConversationStore, Depends(get_conversation_store)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    agent_id: str = DEFAULT_AGENT,
) -> ChatHistory:
    """
    Get chat history for a thread and agent.

    If agent_id is not provided, the default agent will be used.
    """
    sample = await run_in_threadpool(conversations.get_showcase)
    is_public_sample = (
        sample is not None
        and input.thread_id == sample.thread_id
        and input.user_id == sample.user_id
    )
    if not is_public_sample:
        assert_session_owner(input.user_id, chat_identity)
    agent: AgentGraph = get_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": input.thread_id})
    workspace = await run_in_threadpool(conversations.get, input.thread_id)
    if await run_in_threadpool(conversations.is_deleted, input.thread_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    if workspace is not None and (
        workspace.agent_id != agent_id
        or (input.user_id is not None and workspace.user_id != input.user_id)
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        messages: list[BaseMessage] = []
        # Functional-API agents keep the conversation in `__previous__`, which aget_state
        # doesn't return, so read the raw checkpoint first and only fall back for graphs.
        checkpointer = getattr(agent, "checkpointer", None)
        if checkpointer:
            tup = await checkpointer.aget_tuple(config)
            if (
                strict_identity_enabled()
                and not is_public_sample
                and workspace is None
                and tup is None
            ):
                return ChatHistory(messages=[], conversation=None)
            if tup and strict_identity_enabled() and not is_public_sample:
                if tup.metadata.get("user_id") != chat_identity:
                    raise HTTPException(status_code=404, detail="Conversation not found")
            elif tup and input.user_id is not None:
                metadata = tup.metadata
                if metadata.get("user_id") not in (None, input.user_id):
                    raise HTTPException(status_code=404, detail="Conversation not found")
            if tup and "__previous__" in (tup.checkpoint.get("channel_values") or {}):
                messages = messages_from_checkpoint(tup.checkpoint)
        elif (
            strict_identity_enabled()
            and not is_public_sample
            and workspace is None
        ):
            return ChatHistory(messages=[], conversation=None)
        if not messages:
            state_snapshot = await agent.aget_state(config=config)
            messages = state_snapshot.values.get("messages", [])
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages, conversation=workspace)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.get("/{agent_id}/threads", operation_id="threads_with_agent_id")
@router.get("/threads")
async def threads(
    conversations: Annotated[ConversationStore, Depends(get_conversation_store)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    input: UserThreadsInput = Depends(),
    agent_id: str = DEFAULT_AGENT,
) -> UserThreads:
    """
    List a user's conversation threads for an agent, most recently updated first.

    In account mode, the caller's user_id must match the active login session.
    Legacy local mode retains its original caller-supplied identity behavior.
    """
    assert_session_owner(input.user_id, chat_identity)
    agent: AgentGraph = get_agent(agent_id)
    checkpointer = getattr(agent, "checkpointer", None)
    try:
        summaries = (
            await list_user_threads(checkpointer, input.user_id, agent_id, input.limit)
            if checkpointer
            else []
        )
        workspaces = await run_in_threadpool(
            conversations.list, input.user_id, agent_id, input.limit
        )
        summaries = merge_thread_summaries(summaries, workspaces, input.limit)
        deleted = await run_in_threadpool(
            conversations.deleted_thread_ids, [summary.thread_id for summary in summaries]
        )
        summaries = [summary for summary in summaries if summary.thread_id not in deleted]
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

    return UserThreads(threads=summaries)


@router.delete("/conversations/{thread_id}", status_code=204)
async def delete_conversation(
    thread_id: str,
    user_id: Annotated[str, Query(min_length=1, max_length=200)],
    conversations: Annotated[ConversationStore, Depends(get_conversation_store)],
    papers: Annotated[PaperStore, Depends(get_paper_store)],
    guest_id: Annotated[str, Depends(get_guest_identity)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
) -> Response:
    """Stop its work and delete its private chat and exclusively referenced paper."""
    assert_session_owner(user_id, chat_identity)
    if not thread_id or len(thread_id) > 200:
        raise HTTPException(status_code=422, detail="Invalid thread ID")
    if await run_in_threadpool(conversations.is_deleted, thread_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    showcase = await run_in_threadpool(conversations.get_showcase)
    if showcase and showcase.thread_id == thread_id:
        raise HTTPException(status_code=403, detail="Sample conversation is read-only")

    workspace = await run_in_threadpool(conversations.get, thread_id)
    checkpointer = getattr(get_agent(DEFAULT_AGENT), "checkpointer", None)
    checkpoint = (
        await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if checkpointer
        else None
    )
    if workspace is not None:
        if workspace.user_id != user_id or workspace.agent_id != DEFAULT_AGENT:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if checkpoint and (
            checkpoint.metadata.get("user_id") not in (None, user_id)
            or checkpoint.metadata.get("agent_id") not in (None, DEFAULT_AGENT)
        ):
            raise HTTPException(status_code=404, detail="Conversation not found")
    elif (
        not checkpoint
        or checkpoint.metadata.get("user_id") != user_id
        or checkpoint.metadata.get("agent_id") != DEFAULT_AGENT
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        document_ids = _checkpoint_document_ids(checkpoint)
        if workspace is not None and workspace.document_id is not None:
            document_ids.add(workspace.document_id)
        if len(document_ids) > 1:
            raise HTTPException(
                status_code=409,
                detail="Conversation references multiple PDFs; cannot safely delete all of them",
            )
        document_id = next(iter(document_ids), None)
        if document_id is not None:
            if not access.owns(guest_id, document_id):
                raise HTTPException(status_code=404, detail="Conversation not found")
            if access.has_other_owner(guest_id, document_id):
                raise HTTPException(
                    status_code=409,
                    detail="This PDF is also owned by another visitor; it cannot be deleted here",
                )
            if showcase and showcase.document_id == document_id:
                raise HTTPException(status_code=409, detail="This PDF is used by the tutorial")
            other_threads = await run_in_threadpool(
                conversations.other_threads_using_document, document_id, thread_id
            )
            if other_threads:
                raise HTTPException(
                    status_code=409,
                    detail="This PDF is attached to another conversation; unlink it there first",
                )
        await _stop_chat_tasks(thread_id)
        if document_id is not None and (
            await run_in_threadpool(papers.get, document_id) is not None
            or await run_in_threadpool(papers.deletion_state, document_id) == "deleting"
        ):
            await run_in_threadpool(papers.request_deletion, document_id)
            await run_in_threadpool(papers.finish_deletion, document_id)
        if checkpointer:
            await checkpointer.adelete_thread(thread_id)
        await run_in_threadpool(conversations.delete, thread_id, user_id, DEFAULT_AGENT)
        if document_id is not None:
            await run_in_threadpool(access.revoke, guest_id, document_id)
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    finally:
        if not _active_chat_tasks.get(thread_id):
            _deleting_chat_threads.discard(thread_id)
    return Response(status_code=204)


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
