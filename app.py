import os
import json
import logging
import tempfile
import threading
import uuid
from flask import Flask, render_template, session, redirect, request, flash, send_from_directory

import kite_service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── In-memory job store for background backtest runs ─────────────────────────
_backtest_jobs: dict = {}
_jobs_lock = threading.Lock()

_JOBS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__) or "."),
                          "backtest_results", "jobs")
os.makedirs(_JOBS_DIR, exist_ok=True)
log.info("Job store directory: %s", _JOBS_DIR)


def _persist_job(job_id: str, job: dict) -> None:
    """Write job state to disk atomically (rename trick) so it survives a worker restart."""
    path = os.path.join(_JOBS_DIR, f"{job_id}.json")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=_JOBS_DIR, delete=False, suffix=".tmp"
        ) as f:
            tmp_path = f.name
            json.dump(job, f, default=str)
        os.replace(tmp_path, path)          # atomic on POSIX / Windows
        log.info("[job:%s] persisted status=%s to %s", job_id, job.get("status"), path)
    except Exception as exc:
        log.error("[job:%s] persist FAILED: %s", job_id, exc)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _load_job_from_disk(job_id: str) -> dict | None:
    """Read a previously persisted job from disk."""
    path = os.path.join(_JOBS_DIR, f"{job_id}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                job = json.load(f)
            log.info("[job:%s] loaded from disk, status=%s", job_id, job.get("status"))
            return job
    except Exception as exc:
        log.error("[job:%s] disk load FAILED: %s", job_id, exc)
    return None

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
    import backtest_fetcher, time
    access_token = session.get("accessToken")
    if not access_token:
        return {"status": "error", "message": "Not logged in"}, 401
    symbol = request.get_json().get("symbol")
    config = backtest_fetcher.load_stocks()
    stock  = next((s for s in config["stocks"] if s["symbol"] == symbol), None)
    if not stock:
        return {"status": "error", "message": "Stock not found"}, 404

    last_error = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(attempt + 1)   # 2 s, 3 s backoff on retries
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
            last_error = e

    return {"status": "error", "message": str(last_error)}, 500


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
    import time as _time
    initial_job = {"status": "running", "result": None, "error": None,
                   "started_at": _time.time()}
    with _jobs_lock:
        _backtest_jobs[job_id] = initial_job
    # Write immediately so disk has the job even if worker is killed mid-run
    _persist_job(job_id, initial_job)

    def _run():
        log.info("[job:%s] thread started (optimize=%s, n_trials=%s, stocks=%d)",
                 job_id, do_optimize, n_trials, len(stocks_cfg))
        try:
            from backtest.main import run_all_pipeline
            result = run_all_pipeline(
                stocks_cfg      = stocks_cfg,
                optimize_params = do_optimize,
                n_trials        = n_trials,
                default_params  = params_dict if not do_optimize else None,
            )
            if "error" in result:
                log.warning("[job:%s] pipeline returned error: %s", job_id, result["error"])
                job = {"status": "error", "result": None, "error": result["error"]}
                with _jobs_lock:
                    _backtest_jobs[job_id] = job
                _persist_job(job_id, job)
                return
            # Validate JSON-serialisability; sanitise any stray numpy/pandas types
            try:
                json.dumps(result)
            except (TypeError, ValueError) as ser_err:
                log.warning("[job:%s] result had non-serialisable types (%s); sanitising", job_id, ser_err)
                result = json.loads(json.dumps(result, default=str))
            log.info("[job:%s] DONE — per_stock=%d", job_id, len(result.get("per_stock", [])))
            job = {"status": "done", "result": result, "error": None}
            with _jobs_lock:
                _backtest_jobs[job_id] = job
            _persist_job(job_id, job)
        except Exception as exc:
            import traceback; traceback.print_exc()
            log.error("[job:%s] thread EXCEPTION: %s", job_id, exc)
            job = {"status": "error", "result": None, "error": str(exc)}
            with _jobs_lock:
                _backtest_jobs[job_id] = job
            _persist_job(job_id, job)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "success", "job_id": job_id}


_JOB_STALE_SECONDS = 1800  # 30 min — per-stock optimisation runs longer than cross-stock


@app.route("/backtest/strategy/job/<job_id>")
def backtest_strategy_job(job_id):
    """Poll a running or completed backtest job."""
    import time as _time
    with _jobs_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        # Worker may have restarted — try disk fallback
        job = _load_job_from_disk(job_id)
    if not job:
        log.warning("[job:%s] poll returned 404 — not in memory or disk (jobs dir: %s)", job_id, _JOBS_DIR)
        return {"status": "error", "message": "Job not found"}, 404
    # If job is still "running" from a previous (dead) worker, fail it
    if job.get("status") == "running":
        age = _time.time() - job.get("started_at", _time.time())
        if age > _JOB_STALE_SECONDS:
            job = {"status": "error", "result": None,
                   "error": "Job timed out — server was restarted mid-run. Please run again."}
            with _jobs_lock:
                _backtest_jobs[job_id] = job
            _persist_job(job_id, job)
    return {"status": "success", "job": job}


@app.route("/backtest/analysis/groq", methods=["POST"])
def backtest_groq_analysis():
    """Run post-backtest AI analysis on a completed job result."""
    if not session.get("accessToken"):
        return {"status": "error", "message": "Not logged in"}, 401

    data   = request.get_json()
    job_id = data.get("job_id")

    with _jobs_lock:
        job = _backtest_jobs.get(job_id)

    if not job or job.get("status") != "done":
        return {"status": "error", "message": "Job not found or not complete"}, 404

    try:
        import groq_analysis
        result   = job["result"]
        analysis = groq_analysis.analyse_backtest(result)
        return {"status": "success", "analysis": analysis}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}, 500


@app.route("/backtest/stocks/data-status")
def backtest_stocks_data_status():
    """Return server-side data status for all configured stocks (reads actual CSV files)."""
    import backtest_fetcher
    config = backtest_fetcher.load_stocks()
    result = []
    for s in config["stocks"]:
        status = backtest_fetcher.get_data_status(s["symbol"], s["exchange"])
        result.append({
            "symbol":   s["symbol"],
            "exchange": s["exchange"],
            "has_data": status["exists"] and status["rows"] > 0,
            "rows":     status["rows"],
            "from_date": status["from_date"],
            "to_date":   status["to_date"],
        })
    return {"status": "success", "stocks": result}


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
