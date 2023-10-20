import collections
import copy
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.datamodel import (
    EmbeddingRequest,
    EmbeddingReply,
    InfoReply,
    EmbeddingTypeEnum,
    CompletionRequest,
    ChatMessage,
)
from app.settings import embedding_model, timing_decorator

if TYPE_CHECKING:
    from app.embed_service import EmbeddingService
    from app.llm_service import LLMService

app = FastAPI()
srv_embed: Optional["EmbeddingService"] = None
srv_llm: Optional["LLMService"] = None

base_config = {"do_sample": True, "max_new_tokens": 200}


@app.on_event("startup")
@timing_decorator
def startup():
    global srv_embed, srv_llm
    from app.embed_service import EmbeddingService
    from app.llm_service import LLMService

    srv_embed = EmbeddingService()
    srv_llm = LLMService()


@app.get("/info")
def info() -> InfoReply:
    return InfoReply(embedding_model=embedding_model)


@app.get("/")
def index():
    with open("app/chat.html") as fp:
        return HTMLResponse(fp.read())


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
    config = copy.copy(base_config)
    config.update(completion.config)
    return srv_llm.complete([d.model_dump() for d in completion.messages], **config)


@app.exception_handler(Exception)
def handle_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"message": str(exc)})


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()

    conversation = collections.deque(maxlen=20)

    try:
        while True:
            user_text = await ws.receive_text()
            conversation.append(ChatMessage(role="user", content=user_text))
            response = await srv_llm.complete_async(
                [d.model_dump() for d in conversation], **base_config
            )
            conversation.append(ChatMessage(role="assistant", content=response))
            await ws.send_text(response)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
