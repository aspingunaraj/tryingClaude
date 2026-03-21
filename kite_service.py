import hashlib
import requests

API_KEY = "9a4fnxekfbrw0k6d"
API_SECRET = "d4avfhwjneq1kkxqhfj06rfcyhgykng4"
KITE_BASE_URL = "https://api.kite.trade"


def get_login_url():
    return f"https://kite.zerodha.com/connect/login?api_key={API_KEY}&v=3"


def generate_session(request_token):
    checksum = hashlib.sha256(
        f"{API_KEY}{request_token}{API_SECRET}".encode()
    ).hexdigest()
    response = requests.post(
        f"{KITE_BASE_URL}/session/token",
        headers={"X-Kite-Version": "3"},
        data={"api_key": API_KEY, "request_token": request_token, "checksum": checksum},
    )
    body = response.json()
    if body.get("status") == "success":
        return body["data"]
    raise Exception(f"Failed to generate session: {body}")


def get_funds(access_token):
    response = requests.get(
        f"{KITE_BASE_URL}/user/margins",
        headers={
            "X-Kite-Version": "3",
            "Authorization": f"token {API_KEY}:{access_token}",
        },
    )
    body = response.json()
    if body.get("status") == "success":
        return body["data"]
    raise Exception(f"Failed to fetch funds: {body}")
