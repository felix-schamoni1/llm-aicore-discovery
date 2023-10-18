import pytest
from app.datamodel import (
    EmbeddingRequest,
    EmbeddingTypeEnum,
    CompletionRequest,
    ChatMessage,
)


@pytest.fixture(scope="session")
def test_client():
    from srv import app, startup
    from fastapi.testclient import TestClient

    startup()
    return TestClient(app)


def test_embed_all(test_client):
    test_client.get("/").json()

    for e_type in [
        EmbeddingTypeEnum.DEFAULT,
        EmbeddingTypeEnum.QUERY,
        EmbeddingTypeEnum.DOCUMENT,
    ]:
        test_client.post(
            "/embed",
            json=EmbeddingRequest(
                embedding_type=e_type,
                documents=["Hello World", "Goodbye world"],
            ).model_dump(),
        ).json()

    print(
        test_client.post(
            "/complete",
            json=CompletionRequest(
                messages=[
                    ChatMessage(
                        role="user", content="What is your favourite condiment?"
                    ),
                ],
                config={"max_new_tokens": 100},
            ).model_dump(),
        ).json()
    )
