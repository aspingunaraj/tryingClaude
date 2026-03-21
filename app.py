import os
from flask import Flask, render_template, session, redirect, request, flash

import kite_service

app = Flask(__name__)
# Set SECRET_KEY env var in production; this default is for local dev only
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")


def _mask_token(token):
    if not token or len(token) < 8:
        return "****"
    return token[:4] + "..." + token[-4:]


def _format_currency(value):
    if value is None:
        return None
    try:
        return f"\u20b9{float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)


@app.route("/")
def index():
    access_token = session.get("accessToken")
    ctx = {}

    if access_token:
        ctx["loggedIn"] = True
        ctx["userName"]  = session.get("userName", "")
        ctx["userEmail"] = session.get("userEmail", "")
        ctx["userId"]    = session.get("userId", "")
        ctx["broker"]    = session.get("broker", "")
        ctx["loginTime"] = session.get("loginTime", "")
        ctx["accessToken"] = _mask_token(access_token)

        try:
            funds = kite_service.get_funds(access_token)
            equity = funds.get("equity", {})
            ctx["equityNet"]  = _format_currency(equity.get("net"))
            ctx["equityCash"] = _format_currency(
                equity.get("available", {}).get("cash")
            )
        except Exception:
            ctx["equityNet"] = None
            ctx["equityCash"] = None
    else:
        ctx["loggedIn"] = False
        ctx["loginUrl"] = kite_service.get_login_url()

    return render_template("index.html", **ctx)


@app.route("/redirect")
def handle_redirect():
    request_token = request.args.get("request_token")
    status = request.args.get("status", "success")

    if status != "success" or not request_token:
        flash("Login was cancelled or failed. Please try again.", "error")
        return redirect("/")

    try:
        data = kite_service.generate_session(request_token)
        session["accessToken"] = data.get("access_token")
        session["userName"]    = data.get("user_name", "")
        session["userEmail"]   = data.get("email", "")
        session["userId"]      = data.get("user_id", "")
        session["broker"]      = data.get("broker", "")
        session["loginTime"]   = str(data.get("login_time", ""))
    except Exception as e:
        flash(f"Authentication failed: {e}", "error")

    return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ── Backtest module ──────────────────────────────────────────────────────────

@app.route("/backtest")
def backtest():
    import backtest_fetcher
    config = backtest_fetcher.load_stocks()
    stocks = []
    for s in config["stocks"]:
        row = dict(s)
        row["status"] = backtest_fetcher.get_data_status(s["symbol"], s["exchange"])
        stocks.append(row)
    return render_template(
        "backtest.html",
        stocks=stocks,
        loggedIn=bool(session.get("accessToken")),
        userName=session.get("userName", ""),
    )


@app.route("/backtest/stocks/add", methods=["POST"])
def backtest_add_stock():
    import backtest_fetcher
    if not session.get("accessToken"):
        return {"status": "error", "message": "Not logged in"}, 401
    data = request.get_json()
    config = backtest_fetcher.load_stocks()
    if any(s["symbol"] == data["symbol"] for s in config["stocks"]):
        return {"status": "error", "message": "Stock already in list"}, 400
    config["stocks"].append({
        "symbol":           data["symbol"],
        "exchange":         data.get("exchange", "NSE"),
        "instrument_token": int(data["instrument_token"]),
        "name":             data.get("name", data["symbol"]),
    })
    backtest_fetcher.save_stocks(config)
    return {"status": "success"}


@app.route("/backtest/stocks/remove", methods=["POST"])
def backtest_remove_stock():
    import backtest_fetcher
    symbol = request.get_json().get("symbol")
    config = backtest_fetcher.load_stocks()
    config["stocks"] = [s for s in config["stocks"] if s["symbol"] != symbol]
    backtest_fetcher.save_stocks(config)
    return {"status": "success"}


@app.route("/backtest/instruments/search")
def backtest_search_instruments():
    access_token = session.get("accessToken")
    if not access_token:
        return {"status": "error", "message": "Not logged in"}, 401
    query    = request.args.get("q", "").strip()
    exchange = request.args.get("exchange", "NSE")
    if not query:
        return {"status": "success", "data": []}
    try:
        results = kite_service.search_instruments(access_token, exchange, query)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@app.route("/backtest/data/fetch", methods=["POST"])
def backtest_fetch_data():
    import backtest_fetcher
    access_token = session.get("accessToken")
    if not access_token:
        return {"status": "error", "message": "Not logged in"}, 401
    symbol = request.get_json().get("symbol")
    config = backtest_fetcher.load_stocks()
    stock  = next((s for s in config["stocks"] if s["symbol"] == symbol), None)
    if not stock:
        return {"status": "error", "message": "Stock not found"}, 404
    try:
        count  = backtest_fetcher.fetch_stock_data(
            access_token,
            stock["instrument_token"],
            stock["symbol"],
            stock["exchange"],
            days=120,
        )
        status = backtest_fetcher.get_data_status(stock["symbol"], stock["exchange"])
        return {"status": "success", "rows": count, "data_status": status}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
