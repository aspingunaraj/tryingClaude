import hashlib
import io
import csv as _csv
import requests

API_KEY = "9a4fnxekfbrw0k6d"
API_SECRET = "d4avfhwjneq1kkxqhfj06rfcyhgykng4"
KITE_BASE_URL = "https://api.kite.trade"

# In-memory instruments cache keyed by exchange (valid for the process lifetime)
_instruments_cache = {}


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


def _fetch_instruments(access_token, exchange):
    """Download the full instruments CSV for an exchange and cache it."""
    response = requests.get(
        f"{KITE_BASE_URL}/instruments/{exchange}",
        headers={
            "X-Kite-Version": "3",
            "Authorization": f"token {API_KEY}:{access_token}",
        },
        timeout=30,
    )
    response.raise_for_status()
    reader = _csv.DictReader(io.StringIO(response.text))
    instruments = list(reader)
    _instruments_cache[exchange] = instruments
    return instruments


def search_instruments(access_token, exchange, query):
    """Search instruments by tradingsymbol prefix/substring. Returns up to 20 EQ matches."""
    instruments = _instruments_cache.get(exchange) or _fetch_instruments(access_token, exchange)
    query = query.upper()
    results = []
    for inst in instruments:
        sym = inst.get("tradingsymbol", "").upper()
        if query in sym and inst.get("instrument_type") == "EQ":
            results.append({
                "instrument_token": inst["instrument_token"],
                "tradingsymbol": inst["tradingsymbol"],
                "name": inst["name"],
                "exchange": inst["exchange"],
            })
        if len(results) >= 20:
            break
    return results


def get_historical_data(access_token, instrument_token, from_date, to_date, interval="minute"):
    """Fetch OHLCV candles. Returns list of [timestamp, open, high, low, close, volume]."""
    response = requests.get(
        f"{KITE_BASE_URL}/instruments/historical/{instrument_token}/{interval}",
        headers={
            "X-Kite-Version": "3",
            "Authorization": f"token {API_KEY}:{access_token}",
        },
        params={
            "from": from_date,
            "to": to_date,
            "continuous": 0,
            "oi": 0,
        },
        timeout=30,
    )
    body = response.json()
    if body.get("status") == "success":
        return body["data"]["candles"]
    raise Exception(f"Historical data error: {body}")
