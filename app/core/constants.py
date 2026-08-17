from pydantic import BaseModel, ConfigDict, Field


CONTENT_TYPE_MAPPING = {
    "text": "text",
    "reasoning_content": "reasoning"
}


class NSProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    chat_agent: str = Field(..., description="Agent to use for this namespace")
    is_singleton: bool = Field(
        ..., description="Whether this namespace allows only one chat instance"
    )
    agent_params: dict[str, str] | None = Field(
        None, description="Agent parameters for chat agent"
    )


CHAT_NS_MAP = {
    "generic": NSProperty(chat_agent="chat", is_singleton=False),
    "beta_report/androidtv": NSProperty(
        chat_agent="beta_report",
        is_singleton=True,
        agent_params={"platform": "AndroidTV"},
    ),
    "beta_report/dishtv": NSProperty(
        chat_agent="beta_report",
        is_singleton=True,
        agent_params={"platform": "DishTV"},
    ),
}
