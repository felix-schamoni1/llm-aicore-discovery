# returns docker hub secret in the correct format for AI Core
import json

value = {
    "auths": {
        "https://index.docker.io/v1/": {
            "username": "<your user-name>",
            "password": "XXX",
        }
    }
}

print(json.dumps({".dockerconfigjson": json.dumps(value)}))
