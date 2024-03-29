# used for final testing of the deployed LLM
# pip install ai-core-sdk
import json

import requests
from ai_core_sdk.ai_core_v2_client import AICoreV2Client

from app.datamodel import CompletionRequest, ChatMessage
from app.settings import timing_decorator

with open("aicore.json", "r", encoding="utf8") as fp:
    cfg = json.load(fp)

url = "<your-deployment-id>"

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
sess.post = timing_decorator(sess.post)

print(
    sess.post(
        url + "complete",
        json=CompletionRequest(
            messages=[ChatMessage(content="What's the story of the Eiffel Tower?")],
            config={"do_sample": True, "max_new_tokens": 20},
        ).model_dump(),
    ).json()
)


print(
    sess.post(
        "<your-server-url>",
        json=CompletionRequest(
            messages=[ChatMessage(content="What's the story of the Eiffel Tower?")],
            config={"do_sample": True, "max_new_tokens": 20},
        ).model_dump(),
    ).json()
)
