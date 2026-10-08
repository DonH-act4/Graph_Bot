from datetime import datetime
from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator
from typing_extensions import TypedDict

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class EvidenceContextInput(BaseModel):
    """References to source blocks selected by the user for one chat turn."""

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    block_ids: list[str] = Field(min_length=1, max_length=12)

    @field_validator("block_ids")
    @classmethod
    def validate_block_ids(cls, block_ids: list[str]) -> list[str]:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("evidence block IDs must be unique")
        if any(not block_id.startswith("blk_") for block_id in block_ids):
            raise ValueError("invalid evidence block ID")
        return block_ids


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent. Defaults to the default model set in the settings of the service.",
        default=None,
        examples=[OpenAIModelName.GPT_5_NANO, AnthropicModelName.HAIKU_45],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )
    evidence_context: EvidenceContextInput | None = Field(
        description="Source blocks selected for an evidence-grounded turn.",
        default=None,
    )
    document_id: str | None = Field(
        description="Current paper for a paper-grounded conversation.",
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationUpdate(BaseModel):
    """Browser workspace state; chat messages remain in LangGraph checkpoints."""

    user_id: str = Field(min_length=1, max_length=200)
    agent_id: str = Field(default="research-assistant", min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=120)
    document_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    document_name: str | None = Field(default=None, max_length=256)
    selected_block_ids: list[str] = Field(default_factory=list, max_length=12)
    draft_message: str = Field(default="", max_length=10000)

    @field_validator("selected_block_ids")
    @classmethod
    def validate_selection(cls, block_ids: list[str]) -> list[str]:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("selected evidence block IDs must be unique")
        if any(not block_id.startswith("blk_") for block_id in block_ids):
            raise ValueError("invalid evidence block ID")
        return block_ids

    @model_validator(mode="after")
    def require_selected_document(self) -> "ConversationUpdate":
        if self.selected_block_ids and self.document_id is None:
            raise ValueError("selected evidence requires a document")
        return self


class ConversationState(ConversationUpdate):
    thread_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime


class Showcase(BaseModel):
    """Published pointer to a real, persisted demonstration conversation."""

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(max_length=1200)
    thread_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=200)


class ChatHistory(BaseModel):
    messages: list[ChatMessage]
    conversation: ConversationState | None = None


class UserThreadsInput(BaseModel):
    """Input for listing a user's conversation threads."""

    user_id: str = Field(
        description="User ID to list threads for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    limit: int = Field(
        description="Maximum number of threads to return.",
        default=20,
        ge=1,
        le=100,
    )


class ThreadSummary(BaseModel):
    """Summary of a single conversation thread."""

    thread_id: str = Field(
        description="Thread ID of the conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_id: str = Field(
        description="Agent this thread was run with.",
        examples=["research-assistant"],
    )
    updated_at: datetime | None = Field(
        description="Timestamp of the most recent checkpoint in this thread.",
        default=None,
        examples=["2024-07-31T20:14:19.804150+00:00"],
    )
    title: str | None = Field(
        description="Title for the thread, derived from the first human message.",
        default=None,
        examples=["What is the weather in Tokyo?"],
    )
    document_id: str | None = None
    document_name: str | None = None
    selected_block_ids: list[str] = Field(default_factory=list)


class UserThreads(BaseModel):
    threads: list[ThreadSummary]
