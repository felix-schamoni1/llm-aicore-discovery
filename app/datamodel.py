from typing import List, Dict, Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class CompletionRequest(BaseModel):
    messages: List[ChatMessage]
    config: Dict[str, Any] = {}
    # transformers.generation.configuration_utils.GenerationConfig

    def __init__(self, **kwargs):
        if "config" in kwargs:
            for key in ["temperature"]:
                if key in kwargs["config"]:
                    kwargs["config"][key] = float(kwargs["config"][key])

        super().__init__(**kwargs)
