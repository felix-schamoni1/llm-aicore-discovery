import pytest

from app.datamodel import (
    CompletionRequest,
    ChatMessage,
)


req = CompletionRequest(
    messages=[
        ChatMessage(role="user", content="What is your favourite condiment?"),
    ],
    config={"max_new_tokens": 2},
)


@pytest.fixture(scope="session")
def test_client():
    from srv import app, startup
    from fastapi.testclient import TestClient

    startup()
    return TestClient(app)


def test_completion(test_client):
    print(
        test_client.post(
            "/complete",
            json=req.model_dump(),
        ).json()
    )

