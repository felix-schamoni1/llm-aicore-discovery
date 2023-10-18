from enum import Enum
from typing import List, Dict, Any

from pydantic import BaseModel


class InfoReply(BaseModel):
    embedding_model: str


class EmbeddingTypeEnum(int, Enum):
    DEFAULT = 0
    QUERY = 1
    DOCUMENT = 2


class EmbeddingRequest(BaseModel):
    embedding_type: EmbeddingTypeEnum = EmbeddingTypeEnum.DEFAULT
    documents: List[str] = []


class EmbeddingReply(BaseModel):
    embeddings: List[List[float]]


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class CompletionRequest(BaseModel):
    messages: List[ChatMessage]
    config: Dict[str, Any] = {}
    # transformers.generation.configuration_utils.GenerationConfig
