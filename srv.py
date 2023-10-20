from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI
from app.datamodel import (
    EmbeddingRequest,
    EmbeddingReply,
    InfoReply,
    EmbeddingTypeEnum,
    CompletionRequest,
)
from app.settings import embedding_model, timing_decorator

if TYPE_CHECKING:
    from app.embed_service import EmbeddingService
    from app.llm_service import LLMService

app = FastAPI()
srv_embed: Optional["EmbeddingService"] = None
srv_llm: Optional["LLMService"] = None


@app.on_event("startup")
@timing_decorator
def startup():
    global srv_embed, srv_llm
    from app.embed_service import EmbeddingService
    from app.llm_service import LLMService

    srv_embed = EmbeddingService()
    srv_llm = LLMService()


@app.get("/")
def index() -> InfoReply:
    return InfoReply(embedding_model=embedding_model)


@app.post("/embed")
@timing_decorator
def embed(data: EmbeddingRequest) -> EmbeddingReply:
    if not data.documents:
        return EmbeddingReply(embeddings=[])

    if data.embedding_type == EmbeddingTypeEnum.DEFAULT:
        fn = srv_embed.embed
    elif data.embedding_type == EmbeddingTypeEnum.QUERY:
        fn = srv_embed.embed_query
    elif data.embedding_type == EmbeddingTypeEnum.DOCUMENT:
        fn = srv_embed.embed_document
    else:
        raise ValueError(f"Unknown embedding type {data.embedding_type}")

    return EmbeddingReply(embeddings=fn(data.documents).tolist())


@app.post("/complete")
@timing_decorator
def complete(completion: CompletionRequest) -> str:
    return srv_llm.complete(
        [d.model_dump() for d in completion.messages], **completion.config
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
