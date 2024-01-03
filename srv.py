import copy
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI


from app.datamodel import (
    CompletionRequest,
)
from app.settings import timing_decorator, http_prefix

if TYPE_CHECKING:
    from app.llm_service import LLMService

app = FastAPI()
srv_llm: Optional["LLMService"] = None

base_config = {"do_sample": True, "max_new_tokens": 200}


@app.on_event("startup")
@timing_decorator
def startup():
    global srv_llm
    from app.llm_service import LLMService

    srv_llm = LLMService()


@app.post(f"{http_prefix}/complete")
@timing_decorator
def complete(completion: CompletionRequest) -> str:
    config = copy.copy(base_config)
    config.update(completion.config)
    res = list(srv_llm.complete(completion.messages, **config))
    return res[-2]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
