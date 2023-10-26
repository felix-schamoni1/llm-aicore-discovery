# pip install ai-core-sdk
import json

import requests
from ai_core_sdk.ai_core_v2_client import AICoreV2Client

from app.datamodel import EmbeddingRequest, CompletionRequest, ChatMessage

with open("aicore.json", "r", encoding="utf8") as fp:
    cfg = json.load(fp)

url = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d0c1bfa3b92694c9/v1/"

client = AICoreV2Client(
    base_url=cfg["serviceurls"]["AI_API_URL"],
    auth_url=cfg["url"] + "/oauth/token",
    client_id=cfg["clientid"],
    client_secret=cfg["clientsecret"],
    resource_group="default",
)

sess = requests.Session()
sess.headers.update(
    {"Authorization": client.rest_client.get_token(), "AI-Resource-Group": "default"}
)

print(
    sess.post(
        url + "embed", json=EmbeddingRequest(documents=["hello World"]).model_dump()
    ).json()
)
print(
    sess.post(
        url + "complete",
        json=CompletionRequest(
            messages=[ChatMessage(content="What's the story of the Eiffel Tower?")],
            config={
                "do_sample": True,
                "max_new_tokens": 1000
            }
        ).model_dump(),
    ).json()
)
