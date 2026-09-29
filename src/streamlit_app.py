import asyncio
import os
import re
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit_cytoscape import EdgeStyle, Event, NodeStyle, streamlit_cytoscape

from client import (
    AgentClient,
    AgentClientError,
    EvidenceBlockPage,
    GraphArtifact,
    GraphReviews,
    GraphStatus,
    PaperStatus,
)
from schema import ChatHistory, ChatMessage, UserThreads
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"
DOCUMENT_ID_PARAM = "document_id"
_DOCUMENT_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


def restore_paper_from_query_params() -> tuple[bool, str | None]:
    """Restore one existing paper without starting parsing or graph extraction."""
    document_id = st.query_params.get(DOCUMENT_ID_PARAM)
    current_document_id = st.session_state.get("paper_id")
    if document_id is None:
        if current_document_id:
            st.query_params[DOCUMENT_ID_PARAM] = current_document_id
        return False, None
    if not _DOCUMENT_ID_RE.fullmatch(document_id):
        return False, "The document_id URL parameter must be a 64-character lowercase SHA-256."
    if current_document_id == document_id:
        return False, None

    st.session_state.paper_id = document_id
    st.session_state.paper_block_offset = 0
    st.session_state.paper_graph_requested = True
    st.session_state.pop("paper_graph_version", None)
    st.session_state.pop("paper_graph_selection", None)
    return True, None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_threads_cached(
    base_url: str, user_id: str, agent_id: str | None = None, limit: int = 20
) -> UserThreads:
    """
    Fetch and cache user threads using the new synchronous get_user_threads method.
    """
    client = AgentClient(base_url=base_url, get_info=False)
    return client.get_user_threads(user_id=user_id, agent=agent_id, limit=limit)


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # Initialize voice manager (once per session)
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            # Read the agent from the URL so history is fetched through the graph that
            # created the thread.
            resume_agent = st.query_params.get("agent") or agent_client.agent
            try:
                messages: ChatHistory = agent_client.get_history(
                    thread_id=thread_id, agent=resume_agent
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Keep thread_id in the URL so the address bar is directly shareable.
    st.query_params["thread_id"] = st.session_state.thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # Clear saved audio when starting new chat
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.expander(":material/history: Previous Chats", expanded=False):
            try:
                url_agent = st.query_params.get("agent")
                if url_agent in [a.key for a in agent_client.info.agents]:
                    agent_client.agent = url_agent
                else:
                    agent_client.agent = agent_client.info.default_agent
                user_threads = fetch_user_threads_cached(
                    base_url=agent_client.base_url,
                    user_id=user_id,
                    agent_id=agent_client.agent,
                    limit=20,
                )
                thread_list = user_threads.threads
            except Exception as e:
                st.caption(f"Couldn't load conversation history: {e}")
                thread_list = []

            for t in thread_list:
                label = t.title or f"Chat {t.thread_id[:8]}"
                if st.button(label, key=f"thread_{t.thread_id}", use_container_width=True):
                    try:
                        history: ChatHistory = agent_client.get_history(
                            thread_id=t.thread_id, agent=t.agent_id
                        )
                    except AgentClientError:
                        st.error("Could not load that conversation.")
                        continue
                    st.session_state.messages = history.messages
                    st.session_state.thread_id = t.thread_id
                    st.query_params["thread_id"] = t.thread_id
                    if "last_audio" in st.session_state:
                        del st.session_state.last_audio
                    st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            # Sync the selection to the ?agent= URL param (dropped when it's the default).
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
                key="agent",
                bind="query-params",
                on_change=fetch_user_threads_cached.clear,
            )
            use_streaming = st.toggle("Stream results", value=True)
            # Audio toggle with callback: clears cached audio when toggled off
            enable_audio = st.toggle(
                "Enable audio generation",
                value=True,
                disabled=not voice or not voice.tts,
                help="Configure VOICE_TTS_PROVIDER in .env to enable"
                if not voice or not voice.tts
                else None,
                on_change=lambda: (
                    st.session_state.pop("last_audio", None)
                    if not st.session_state.get("enable_audio", True)
                    else None
                ),
                key="enable_audio",
            )

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            # st.context.url is the browser URL (with query string stripped). Rebuild
            # the params, including the agent so the thread resumes through the right graph.
            if not st.context.url:
                st.error("Could not determine the app URL to build a shareable link.")
                return
            query = urllib.parse.urlencode(
                {
                    "thread_id": st.session_state.thread_id,
                    "agent": agent_client.agent,
                    USER_ID_COOKIE: user_id,
                }
            )
            chat_url = f"{st.context.url}?{query}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    draw_paper_upload(agent_client)

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Render saved audio for the last AI message (if it exists)
    # This ensures audio persists across st.rerun() calls
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # Generate new message if the user provided new input
    # Use voice manager if available, otherwise fall back to regular input
    # REQUIRED: Set VOICE_STT_PROVIDER, VOICE_TTS_PROVIDER, OPENAI_API_KEY
    # in app .env (NOT service .env) to enable voice features.
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        is_first_message = len(messages) == 0
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # Generate TTS audio for streaming response
                # Note: draw_messages() stores the final message in st.session_state.messages
                # and the container reference in st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # Only generate audio for AI responses with content
                    if last_msg.type == "ai" and last_msg.content:
                        # Use audio_only=True since text was already streamed by draw_messages()
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # Render AI response with optional voice
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            if is_first_message:
                fetch_user_threads_cached.clear()
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


def draw_paper_upload(agent_client: AgentClient) -> None:
    """Show the limited phase 1 PDF workflow without claiming a graph exists yet."""
    # UI guard only; the API independently enforces this exact-sample boundary.
    max_sample_bytes = 2_215_244
    has_paper_context = bool(
        st.session_state.get("paper_id") or st.query_params.get(DOCUMENT_ID_PARAM)
    )
    with st.expander(
        "EvidenceGraph · internal PDF sample", expanded=has_paper_context
    ):
        restored_from_url, restore_error = restore_paper_from_query_params()
        st.caption(
            "Only the three exact phase 0 sample PDFs are accepted. "
            "Parsing does not require an LLM API key."
        )
        if restore_error:
            st.error(restore_error)
        uploaded = st.file_uploader("Choose a sample PDF", type=["pdf"], key="paper_pdf")
        too_large = uploaded is not None and uploaded.size > max_sample_bytes
        if too_large:
            st.error("This PDF exceeds the internal sample size limit.")
        if st.button("Upload PDF", disabled=uploaded is None or too_large, key="paper_upload"):
            try:
                result = agent_client.upload_paper(uploaded.getvalue())
                st.session_state.paper_id = result.document_id
                st.session_state.paper_block_offset = 0
                st.session_state.paper_graph_requested = False
                st.session_state.pop("paper_graph_version", None)
                st.session_state.pop("paper_graph_selection", None)
                st.query_params[DOCUMENT_ID_PARAM] = result.document_id
            except AgentClientError as exc:
                st.error(str(exc))

        document_id = st.session_state.get("paper_id")
        if not document_id:
            return
        st.caption(f"Document version: {document_id}")
        st.button("Refresh processing status", key="paper_refresh")
        try:
            status: PaperStatus = agent_client.get_paper(document_id)
        except AgentClientError as exc:
            if restored_from_url:
                st.error(f"Could not restore the document from this URL: {exc}")
            else:
                st.error(str(exc))
            return
        if status.state == "processing":
            st.info("PDF is being parsed. Refresh to check again.")
        elif status.state == "ready":
            st.success(
                f"Parsing complete: {status.page_count} pages, {status.block_count} evidence blocks."
            )
            draw_candidate_graph(agent_client, document_id)
            with st.expander("Browse all parsed evidence blocks", expanded=False):
                draw_evidence_blocks(agent_client, document_id)
        elif status.state == "failed":
            st.error(status.error or "PDF processing failed. Upload the sample again to retry.")
        else:
            st.warning("Unknown processing state returned by the service.")


def draw_candidate_graph(agent_client: AgentClient, document_id: str) -> None:
    """Control extraction and show the first auditable graph representation."""
    st.markdown("#### Candidate evidence graph")
    st.caption(
        "Machine-extracted nodes and relationships remain candidates until human review."
    )
    if st.button("Generate candidate graph", key="paper_graph_generate"):
        try:
            agent_client.request_paper_graph(document_id)
            st.session_state.paper_graph_requested = True
        except AgentClientError as exc:
            st.error(str(exc))
    if st.button("Refresh graph status", key="paper_graph_refresh"):
        st.session_state.paper_graph_requested = True
    if not st.session_state.get("paper_graph_requested", False):
        return

    try:
        status: GraphStatus = agent_client.get_paper_graph_status(document_id)
    except AgentClientError as exc:
        st.error(str(exc))
        return
    if status.state in {"queued", "processing"}:
        st.info(f"Graph extraction is {status.state}. Refresh to check again.")
        if status.current_version is None:
            return
    elif status.state == "failed":
        st.error(status.error or "Graph extraction failed. Generate again to retry.")
        if status.current_version is None:
            return
    elif status.state == "ready":
        st.success(
            f"Candidate graph ready: {status.node_count} nodes, "
            f"{status.relation_count} relationships."
        )

    try:
        versions = agent_client.get_paper_graph_versions(document_id)
    except AgentClientError as exc:
        st.error(str(exc))
        return
    version_numbers = [item.version for item in versions.versions]
    current_index = version_numbers.index(versions.current_version)
    selected_version = st.selectbox(
        "Graph version",
        version_numbers,
        index=current_index,
        format_func=lambda version: (
            f"v{version} (current)" if version == versions.current_version else f"v{version}"
        ),
        key="paper_graph_version",
    )
    st.caption(
        f"Current graph: v{versions.current_version} · "
        f"{len(versions.versions)} stored version(s)."
    )
    if st.button("Rebuild candidate graph", key="paper_graph_rebuild"):
        try:
            agent_client.rebuild_paper_graph(document_id)
            st.rerun()
        except AgentClientError as exc:
            st.error(str(exc))

    try:
        artifact = agent_client.get_paper_graph(document_id, version=selected_version)
        reviews = agent_client.get_paper_graph_reviews(
            document_id, version=selected_version
        )
    except AgentClientError as exc:
        st.error(str(exc))
        return
    draw_graph_artifact(artifact, reviews)


def draw_graph_artifact(artifact: GraphArtifact, reviews: GraphReviews) -> None:
    """Render an interactive typed network with one evidence detail panel."""
    reviewed = len(reviews.reviews)
    total_relations = len(artifact.graph.relations)
    node_metric, relation_metric, review_metric = st.columns(3)
    node_metric.metric("Nodes", len(artifact.graph.nodes))
    relation_metric.metric("Relationships", total_relations)
    review_metric.metric("Reviewed", f"{reviewed}/{total_relations}")

    with st.expander("Graph run details", expanded=False):
        st.caption(
            f"Graph v{artifact.graph_version} · extractor: {artifact.extractor_name} · "
            f"model/version: {artifact.extractor_version}"
        )
        if artifact.usage is None:
            st.caption("Token usage unavailable for this graph version.")
        else:
            usage = artifact.usage
            st.caption(
                f"Tokens: {usage.input_tokens:,} input · {usage.output_tokens:,} output · "
                f"{usage.thinking_tokens:,} thinking · {usage.total_tokens:,} total"
            )
            if usage.paid_standard_estimate_usd is not None:
                st.caption(
                    f"Paid Standard reference: ${usage.paid_standard_estimate_usd:.6f} USD · "
                    "Free Tier actual charge is $0 while within its limits."
                )
                if usage.pricing_basis:
                    st.caption(f"Pricing snapshot: {usage.pricing_basis}")

    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;gap:.45rem;margin:.2rem 0 .65rem 0">
          <span style="color:#7c3aed">● Paper</span>
          <span style="color:#0ea5e9">● Method</span>
          <span style="color:#22c55e">● Dataset</span>
          <span style="color:#f59e0b">● Result</span>
          <span style="color:#f43f5e">● Claim</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Drag nodes to explore · scroll to zoom · click any node or arrow to inspect its evidence."
    )
    event = streamlit_cytoscape(
        elements=build_graph_elements(artifact, reviews),
        layout=build_graph_layout(),
        node_styles=build_graph_node_styles(),
        edge_styles=build_graph_edge_styles(),
        height=600,
        key=f"paper_graph_network_{artifact.document_sha256}_{artifact.graph_version}",
        events=[
            Event("inspect_node", "tap", "node"),
            Event("inspect_relation", "tap", "edge"),
        ],
    )
    selection = resolve_graph_selection(artifact, event)
    if selection is not None:
        kind, index = selection
        selected_id = (
            artifact.graph.nodes[index].node_id
            if kind == "node"
            else artifact.graph.relations[index].relation_id
        )
        st.session_state.paper_graph_selection = {
            "document_sha256": artifact.document_sha256,
            "graph_version": artifact.graph_version,
            "event": event,
            "selected_id": selected_id,
        }
    else:
        stored = st.session_state.get("paper_graph_selection")
        if (
            isinstance(stored, dict)
            and stored.get("document_sha256") == artifact.document_sha256
            and stored.get("graph_version") == artifact.graph_version
        ):
            selection = resolve_graph_selection(artifact, stored.get("event"))
    draw_graph_inspector(artifact, reviews, selection)


def build_graph_layout() -> dict[str, Any]:
    """Return a stable hub-and-ring layout that also fits narrow screens."""
    return {
        "name": "concentric",
        "animate": False,
        "fit": True,
        "padding": 48,
        "avoidOverlap": True,
        "minNodeSpacing": 20,
        "spacingFactor": 0.9,
    }


def build_graph_elements(
    artifact: GraphArtifact, reviews: GraphReviews | None = None
) -> dict[str, list[dict[str, dict[str, Any]]]]:
    """Convert the audited graph artifact into Cytoscape elements."""
    decision_by_id = {
        review.relation_id: review.decision for review in reviews.reviews
    } if reviews else {}
    nodes = [
        {
            "data": {
                "id": f"node:{node.node_id}",
                "domain_id": node.node_id,
                "label": node.node_type.upper(),
                "name": node.name,
                "type": node.node_type,
                "evidence_count": len(node.evidence),
            }
        }
        for node in artifact.graph.nodes
    ]
    edges = []
    for relation in artifact.graph.relations:
        review_status = decision_by_id.get(relation.relation_id, relation.status)
        edges.append(
            {
                "data": {
                    "id": f"edge:{relation.relation_id}",
                    "domain_id": relation.relation_id,
                    "source": f"node:{relation.source_node_id}",
                    "target": f"node:{relation.target_node_id}",
                    "label": f"RELATION_{review_status.upper()}",
                    "relation_type": relation.relation_type,
                    "review_status": review_status,
                    "rationale": relation.rationale,
                    "evidence_count": len(relation.evidence),
                }
            }
        )
    return {"nodes": nodes, "edges": edges}


def build_graph_node_styles() -> list[NodeStyle]:
    """Use stable semantic colors and circular shapes for graph nodes."""
    visuals = {
        "PAPER": ("#7c3aed", 86),
        "METHOD": ("#0ea5e9", 74),
        "DATASET": ("#22c55e", 68),
        "RESULT": ("#f59e0b", 64),
        "CLAIM": ("#f43f5e", 60),
    }
    return [
        NodeStyle(
            label=node_type,
            color=color,
            caption="name",
            custom_styles={
                "shape": "ellipse",
                "width": size,
                "height": size,
                "border-width": 3,
                "border-color": "#ffffff",
                "color": "#ffffff",
                "font-size": 11,
                "font-weight": 650,
                "text-wrap": "wrap",
                "text-max-width": size - 12,
                "text-valign": "center",
                "text-halign": "center",
                "text-outline-color": color,
                "text-outline-width": 1,
                "shadow-blur": 12,
                "shadow-color": color,
                "shadow-opacity": 0.24,
            },
        )
        for node_type, (color, size) in visuals.items()
    ]


def build_graph_edge_styles() -> list[EdgeStyle]:
    """Make relation review state visible without duplicating it as a list."""
    visuals = {
        "RELATION_CANDIDATE": ("#64748b", "dashed", 3),
        "RELATION_ACCEPTED": ("#16a34a", "solid", 4),
        "RELATION_REJECTED": ("#dc2626", "dotted", 3),
    }
    return [
        EdgeStyle(
            label=status,
            color=color,
            caption="relation_type",
            directed=True,
            curve_style="bezier",
            custom_styles={
                "width": width,
                "line-style": line_style,
                "arrow-scale": 1.1,
                "font-size": 10,
                "font-weight": 600,
                "text-background-opacity": 0.92,
                "text-background-padding": 3,
                "text-background-shape": "round-rectangle",
            },
        )
        for status, (color, line_style, width) in visuals.items()
    ]


def resolve_graph_selection(
    artifact: GraphArtifact, event: object
) -> tuple[str, int] | None:
    """Resolve a Cytoscape event to an audited node or relation index."""
    if not isinstance(event, dict):
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    target_id = data.get("target_id")
    target_group = data.get("target_group")
    if not isinstance(target_id, str):
        return None
    if target_group == "nodes" and target_id.startswith("node:"):
        node_id = target_id.removeprefix("node:")
        for index, node in enumerate(artifact.graph.nodes):
            if node.node_id == node_id:
                return "node", index
    if target_group == "edges" and target_id.startswith("edge:"):
        relation_id = target_id.removeprefix("edge:")
        for index, relation in enumerate(artifact.graph.relations):
            if relation.relation_id == relation_id:
                return "relation", index
    return None


def draw_graph_inspector(
    artifact: GraphArtifact,
    reviews: GraphReviews,
    selection: tuple[str, int] | None,
) -> None:
    """Resolve the clicked graph item to evidence through the API."""
    node_by_id = {node.node_id: node for node in artifact.graph.nodes}
    st.markdown("#### Selected evidence")
    if selection is None:
        st.info("Click a node or relationship in the graph to inspect its source evidence.")
        return
    kind, index = selection
    if kind == "node":
        node = artifact.graph.nodes[index]
        st.markdown(f"**{node.name}** · `{node.node_type}`")
        refs = node.evidence
    else:
        relation = artifact.graph.relations[index]
        decision_by_id = {
            review.relation_id: review.decision for review in reviews.reviews
        }
        decision = decision_by_id.get(relation.relation_id)
        source = node_by_id[relation.source_node_id].name
        target = node_by_id[relation.target_node_id].name
        st.markdown(
            f"**{source} —{relation.relation_type}→ {target}** · "
            f"`{decision or relation.status}`"
        )
        st.caption(relation.rationale)
        refs = relation.evidence

    agent_client: AgentClient = st.session_state.agent_client
    for ref in refs:
        try:
            block = agent_client.get_paper_block(artifact.document_sha256, ref.block_id)
        except AgentClientError as exc:
            st.error(str(exc))
            continue
        pages = sorted({location.page_number for location in block.locations})
        with st.container(border=True):
            st.caption(f"{ref.block_id} · pages {', '.join(map(str, pages))}")
            st.write(block.text)

    if kind == "relation":
        accept, reject = st.columns(2)
        if accept.button(
            "Accept relationship",
            key=f"accept_{relation.relation_id}",
            disabled=decision == "accepted",
        ):
            try:
                agent_client.review_paper_graph_relation(
                    artifact.document_sha256, relation.relation_id, "accepted"
                )
                st.rerun()
            except AgentClientError as exc:
                st.error(str(exc))
        if reject.button(
            "Reject relationship",
            key=f"reject_{relation.relation_id}",
            disabled=decision == "rejected",
        ):
            try:
                agent_client.review_paper_graph_relation(
                    artifact.document_sha256, relation.relation_id, "rejected"
                )
                st.rerun()
            except AgentClientError as exc:
                st.error(str(exc))


def draw_evidence_blocks(agent_client: AgentClient, document_id: str) -> None:
    """Render one small, source-located evidence page for human inspection."""
    page_size = 10
    offset = st.session_state.get("paper_block_offset", 0)
    try:
        page: EvidenceBlockPage = agent_client.get_paper_blocks(
            document_id, offset=offset, limit=page_size
        )
    except AgentClientError as exc:
        st.error(str(exc))
        return

    if page.total == 0:
        st.warning("Parsing completed without displayable evidence blocks.")
        return
    first = page.offset + 1
    last = min(page.offset + len(page.items), page.total)
    st.markdown(f"#### Evidence blocks {first}–{last} of {page.total}")
    for block in page.items:
        pages = sorted({location.page_number for location in block.locations})
        with st.container(border=True):
            st.caption(f"{block.label} · pages {', '.join(map(str, pages))} · {block.block_id}")
            st.write(block.text)
            for location in block.locations:
                box = location.bounding_box
                st.caption(
                    f"Page {location.page_number}: ({box.left:.1f}, {box.top:.1f})–"
                    f"({box.right:.1f}, {box.bottom:.1f}), origin {box.coordinate_origin}"
                )

    previous, following = st.columns(2)
    if previous.button("Previous evidence", disabled=page.offset == 0, key="paper_previous"):
        st.session_state.paper_block_offset = max(0, page.offset - page_size)
        st.rerun()
    if following.button(
        "Next evidence",
        disabled=page.offset + len(page.items) >= page.total,
        key="paper_next",
    ):
        st.session_state.paper_block_offset = page.offset + page_size
        st.rerun()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # Use different labels for transfer vs regular tool calls
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # Expect one ToolMessage for each tool call.
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # Only non-transfer tool calls reach this point
                            status = call_results[tool_call["id"]]
                            status.write("Input:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
