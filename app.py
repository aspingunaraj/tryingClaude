import os
import threading
import uuid
from flask import Flask, render_template, session, redirect, request, flash, send_from_directory

import kite_service

# ── In-memory job store for background backtest runs ─────────────────────────
_backtest_jobs: dict = {}
_jobs_lock = threading.Lock()

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


# ── Backtest strategy routes ─────────────────────────────────────────────────

@app.route("/backtest/strategy/run", methods=["POST"])
def backtest_strategy_run():
    """
    Start an all-stocks backtest job in a background thread.
    Params are universal (cross-stock optimisation or slider values).
    Returns a job_id to poll.
    """
    if not session.get("accessToken"):
        return {"status": "error", "message": "Not logged in"}, 401

    data        = request.get_json()
    do_optimize = bool(data.get("optimize", False))
    n_trials    = int(data.get("n_trials", 50))
    params_dict = data.get("params", {})

    # Load configured stock list
    cfg_path = os.path.join(os.path.dirname(__file__), "backtest_stocks.json")
    try:
        import json as _json
        with open(cfg_path) as f:
            stocks_cfg = _json.load(f).get("stocks", [])
    except Exception:
        stocks_cfg = []

    if not stocks_cfg:
        return {"status": "error", "message": "No stocks configured. Add stocks in the Data tab first."}, 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _backtest_jobs[job_id] = {"status": "running", "result": None, "error": None}

    def _run():
        try:
            from backtest.main import run_all_pipeline
            result = run_all_pipeline(
                stocks_cfg      = stocks_cfg,
                optimize_params = do_optimize,
                n_trials        = n_trials,
                default_params  = params_dict if not do_optimize else None,
            )
            if "error" in result:
                with _jobs_lock:
                    _backtest_jobs[job_id]["status"] = "error"
                    _backtest_jobs[job_id]["error"]  = result["error"]
                return
            with _jobs_lock:
                _backtest_jobs[job_id]["status"] = "done"
                _backtest_jobs[job_id]["result"] = result
        except Exception as exc:
            import traceback; traceback.print_exc()
            with _jobs_lock:
                _backtest_jobs[job_id]["status"] = "error"
                _backtest_jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "success", "job_id": job_id}


@app.route("/backtest/strategy/job/<job_id>")
def backtest_strategy_job(job_id):
    """Poll a running or completed backtest job."""
    with _jobs_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        return {"status": "error", "message": "Job not found"}, 404
    return {"status": "success", "job": job}


@app.route("/backtest/results/chart/<exchange>/<symbol>")
def backtest_result_chart(exchange, symbol):
    """Serve the saved PNG chart for a symbol."""
    filename = f"{exchange}_{symbol}_charts.png"
    results_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
    if not os.path.exists(os.path.join(results_dir, filename)):
        return "Chart not found", 404
    return send_from_directory(results_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
