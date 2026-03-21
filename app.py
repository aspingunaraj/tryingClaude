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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
