import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)


class AgentClientError(Exception):
    pass


class PaperStatus(BaseModel):
    """Status returned by the internal EvidenceGraph PDF endpoint."""

    document_id: str
    state: Literal["processing", "ready", "failed"]
    page_count: int | None = None
    block_count: int | None = None
    error: str | None = None


class EvidenceBoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
    coordinate_origin: str


class EvidenceLocation(BaseModel):
    page_number: int = Field(ge=1)
    bounding_box: EvidenceBoundingBox
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)


class EvidenceBlock(BaseModel):
    block_id: str
    source_ref: str
    label: str
    text: str
    locations: tuple[EvidenceLocation, ...]


class EvidenceBlockPage(BaseModel):
    document_id: str
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=50)
    items: tuple[EvidenceBlock, ...]


class GraphStatus(BaseModel):
    document_id: str
    state: Literal["queued", "processing", "ready", "failed"]
    node_count: int | None = None
    relation_count: int | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    current_version: int | None = None
    error: str | None = None


class GraphEvidenceRef(BaseModel):
    source_sha256: str
    block_id: str


class GraphNode(BaseModel):
    node_id: str
    node_type: Literal["paper", "method", "dataset", "result", "claim"]
    name: str
    document_sha256: str
    evidence: tuple[GraphEvidenceRef, ...]


class GraphRelation(BaseModel):
    relation_id: str
    source_node_id: str
    target_node_id: str
    relation_type: Literal["introduces", "uses", "evaluated_on", "reports", "builds_on"]
    status: Literal["candidate", "unconfirmed"]
    evidence: tuple[GraphEvidenceRef, ...]
    rationale: str


class CandidateGraph(BaseModel):
    schema_version: str
    nodes: tuple[GraphNode, ...]
    relations: tuple[GraphRelation, ...]


class GraphUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    thinking_tokens: int = Field(ge=0)
    tool_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    paid_standard_estimate_usd: float | None = Field(default=None, ge=0)
    pricing_basis: str | None = None


class GraphArtifact(BaseModel):
    schema_version: str
    graph_version: int = Field(default=1, ge=1)
    document_sha256: str
    extractor_name: str
    extractor_version: str
    usage: GraphUsage | None = None
    graph: CandidateGraph


class RelationReview(BaseModel):
    relation_id: str
    decision: Literal["accepted", "rejected"]
    reviewed_at: str


class GraphReviews(BaseModel):
    schema_version: str
    document_sha256: str
    graph_version: int = Field(default=1, ge=1)
    reviews: tuple[RelationReview, ...]


class GraphVersionSummary(BaseModel):
    version: int = Field(ge=1)
    extractor_name: str
    extractor_version: str
    node_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)


class GraphVersions(BaseModel):
    document_id: str
    current_version: int = Field(ge=1)
    versions: tuple[GraphVersionSummary, ...]


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def upload_paper(self, content: bytes) -> PaperStatus:
        """Submit one approved PDF version as a raw application/pdf body."""
        try:
            response = httpx.post(
                f"{self.base_url}/papers",
                content=content,
                headers={**self._headers, "Content-Type": "application/pdf"},
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        return self._parse_paper_status(response)

    def get_paper(self, document_id: str) -> PaperStatus:
        """Read a persisted processing state; the UI refreshes only on user action."""
        try:
            response = httpx.get(
                f"{self.base_url}/papers/{document_id}",
                headers=self._headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        return self._parse_paper_status(response)

    def get_paper_blocks(
        self, document_id: str, *, offset: int = 0, limit: int = 10
    ) -> EvidenceBlockPage:
        """Read one bounded evidence page after PDF parsing succeeds."""
        try:
            response = httpx.get(
                f"{self.base_url}/papers/{document_id}/blocks",
                params={"offset": offset, "limit": limit},
                headers=self._headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return EvidenceBlockPage.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned invalid evidence blocks") from exc

    def get_paper_block(self, document_id: str, block_id: str) -> EvidenceBlock:
        """Read one exact evidence block selected from a graph item."""
        try:
            response = httpx.get(
                f"{self.base_url}/papers/{document_id}/blocks/{block_id}",
                headers=self._headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return EvidenceBlock.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned an invalid evidence block") from exc

    def request_paper_graph(self, document_id: str) -> GraphStatus:
        """Queue candidate extraction after PDF parsing is ready."""
        return self._paper_graph_status_request("POST", f"/papers/{document_id}/graph")

    def rebuild_paper_graph(self, document_id: str) -> GraphStatus:
        """Queue a new immutable version while preserving the current graph."""
        return self._paper_graph_status_request(
            "POST", f"/papers/{document_id}/graph/rebuild"
        )

    def get_paper_graph_status(self, document_id: str) -> GraphStatus:
        """Read candidate extraction state without starting new work."""
        return self._paper_graph_status_request(
            "GET", f"/papers/{document_id}/graph/status"
        )

    def get_paper_graph(
        self, document_id: str, *, version: int | None = None
    ) -> GraphArtifact:
        """Read a validated candidate graph after extraction succeeds."""
        request_options: dict[str, Any] = {}
        if version is not None:
            request_options["params"] = {"version": version}
        try:
            response = httpx.get(
                f"{self.base_url}/papers/{document_id}/graph",
                headers=self._headers,
                timeout=10.0,
                **request_options,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return GraphArtifact.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned an invalid graph") from exc

    def get_paper_graph_versions(self, document_id: str) -> GraphVersions:
        """List immutable graph versions and identify the current one."""
        try:
            response = httpx.get(
                f"{self.base_url}/papers/{document_id}/graph/versions",
                headers=self._headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return GraphVersions.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned invalid graph versions") from exc

    def get_paper_graph_reviews(
        self, document_id: str, *, version: int | None = None
    ) -> GraphReviews:
        """Read the human decisions stored separately from model output."""
        path = f"/papers/{document_id}/graph/reviews"
        if version is not None:
            path = f"{path}?version={version}"
        return self._paper_graph_reviews_request("GET", path)

    def review_paper_graph_relation(
        self, document_id: str, relation_id: str, decision: Literal["accepted", "rejected"]
    ) -> GraphReviews:
        """Accept or reject one candidate relation without rewriting the graph."""
        return self._paper_graph_reviews_request(
            "PUT",
            f"/papers/{document_id}/graph/relations/{relation_id}/review",
            json={"decision": decision},
        )

    def _paper_graph_reviews_request(
        self, method: str, path: str, *, json: dict[str, str] | None = None
    ) -> GraphReviews:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                json=json,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return GraphReviews.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned invalid graph reviews") from exc

    def _paper_graph_status_request(self, method: str, path: str) -> GraphStatus:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentClientError(self._paper_error(exc.response)) from exc
        except httpx.RequestError as exc:
            raise AgentClientError("Paper service is unavailable") from exc
        try:
            return GraphStatus.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned an invalid graph status") from exc

    @staticmethod
    def _parse_paper_status(response: httpx.Response) -> PaperStatus:
        try:
            return PaperStatus.model_validate(response.json())
        except ValueError as exc:
            raise AgentClientError("Paper service returned an invalid status response") from exc

    @staticmethod
    def _paper_error(response: httpx.Response) -> str:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = detail if isinstance(detail, str) else "Request failed"
        return f"Paper request failed ({response.status_code}): {message}"

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            # Don't yield empty string tokens as they cause generator issues
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent: str | None = None) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            agent (str, optional): The agent whose graph should interpret the thread.
        """
        agent = agent or self.agent
        request = ChatHistoryInput(thread_id=thread_id)
        url = f"{self.base_url}/{agent}/history" if agent else f"{self.base_url}/history"
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def _user_threads_request(
        self, user_id: str, agent: str | None, limit: int
    ) -> tuple[str, dict[str, Any]]:
        agent_id = agent or self.agent
        url = f"{self.base_url}/{agent_id}/threads" if agent_id else f"{self.base_url}/threads"
        return url, UserThreadsInput(user_id=user_id, limit=limit).model_dump()

    def get_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """
        List a user's conversation threads.

        Args:
            user_id (str): User ID to list threads for.
            agent (str, optional): The agent whose threads should be listed.
            limit (int, optional): Maximum number of threads to return.
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        try:
            response = httpx.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())

    async def aget_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """
        List a user's conversation threads asynchronously.

        Args:
            user_id (str): User ID to list threads for.
            agent (str, optional): The agent whose threads should be listed.
            limit (int, optional): Maximum number of threads to return.
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())
