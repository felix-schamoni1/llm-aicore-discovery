import json

value = {
    "auths": {
        "https://index.docker.io/v1/": {
            "username": "niklasfruehauf",
            "password": "XXX",
        }
    }
}

print(json.dumps({".dockerconfigjson": json.dumps(value)}))
