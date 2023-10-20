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


def test_completion(test_client):
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


def test_completion_ws(test_client):
    with test_client.websocket_connect("/ws") as sock:
        sock.send_text("Hello, how are you?")
        print(sock.receive_text())
        sock.send_text("Can you tell me on which data you were trained?")
        print(sock.receive_text())
