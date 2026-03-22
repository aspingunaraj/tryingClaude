"""
Post-backtest AI analysis using Groq LLM.

Sends a structured summary of backtest results to llama-3.3-70b-versatile
and returns a plain-text analysis covering weaknesses, patterns, and
actionable improvements.
"""
from __future__ import annotations

import os
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_backtest(result: dict) -> dict:
    """
    Takes the dict returned by run_all_pipeline and returns an analysis dict:
    {
      "status":  "success" | "error",
      "summary": str,          # 2–3 sentence overall take
      "weaknesses": [str],     # 3 bullet points
      "patterns":  [str],      # observed patterns
      "improvements": [str],   # 2–3 specific actionable suggestions
      "raw": str,              # full LLM response text
    }
    """
    try:
        if not GROQ_API_KEY:
            return {"status": "error", "message": "GROQ_API_KEY environment variable not set."}
        prompt = _build_prompt(result)
        raw    = _call_groq(prompt)
        parsed = _parse_response(raw)
        parsed["raw"]    = raw
        parsed["status"] = "success"
        return parsed
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(result: dict) -> str:
    agg        = result.get("aggregate", {})
    ml_agg     = result.get("ml_aggregate", {})
    per_stock  = result.get("per_stock", [])
    params     = result.get("best_params", {})
    n_cfg      = result.get("n_configured", len(per_stock))
    n_loaded   = result.get("n_loaded", len(per_stock))

    pm  = agg.get("portfolio_metrics", {})
    am  = agg.get("avg_metrics", {})
    mlm = ml_agg.get("portfolio_metrics", {})

    # ── Exit reason breakdown ──────────────────────────────────────────────
    exit_reasons: dict = {}
    for s in per_stock:
        for trades_key in ("test_trades",):   # may not be present after stripping
            pass
    # Aggregate exit reasons from test_metrics proxy (n_trades is all we have
    # after DataFrames are stripped — do best effort from per_stock)

    # ── Per-stock table (top 3 / bottom 3 by test Sharpe) ────────────────
    sortable = [s for s in per_stock if s.get("test_metrics")]
    sortable.sort(key=lambda s: s["test_metrics"].get("sharpe_ratio", -999))
    bottom3 = sortable[:3]
    top3    = sortable[-3:][::-1]

    def fmt_stock(s):
        tm = s.get("test_metrics", {})
        return (
            f"  {s['symbol']}: Sharpe={tm.get('sharpe_ratio',0):.2f}, "
            f"Return={tm.get('total_return_pct',0):.2f}%, "
            f"WinRate={tm.get('win_rate_pct',0):.1f}%, "
            f"Trades={tm.get('n_trades',0)}, "
            f"MaxDD={tm.get('max_drawdown_pct',0):.2f}%"
        )

    top3_str    = "\n".join(fmt_stock(s) for s in top3)    or "  —"
    bottom3_str = "\n".join(fmt_stock(s) for s in bottom3) or "  —"

    # ── Strategy params ───────────────────────────────────────────────────
    threshold   = params.get("threshold", 0.005) * 100
    stop_loss   = params.get("stop_loss",  0.003) * 100
    take_profit = params.get("take_profit",0.006) * 100
    vol_filter  = params.get("volume_filter", 1.5)
    max_hold    = params.get("max_holding", 30)
    skip_open   = params.get("time_open_filter", 15)
    skip_close  = params.get("time_close_filter", 15)

    prompt = f"""You are an expert quantitative analyst reviewing intraday backtesting results.

STRATEGY OVERVIEW
-----------------
Type        : VWAP Mean-Reversion (1-minute candles, intraday only)
Universe    : NIFTY 50 stocks (NSE India)
Configured  : {n_cfg} stocks   |   Loaded with data: {n_loaded} stocks

STRATEGY PARAMETERS
-------------------
VWAP deviation threshold : {threshold:.2f}%
Stop loss                : {stop_loss:.2f}%
Take profit              : {take_profit:.2f}%
Volume filter            : {vol_filter:.1f}× rolling average
Max holding time         : {max_hold} minutes
Skip market open         : first {skip_open} minutes
Skip market close        : last {skip_close} minutes

BASE STRATEGY — PORTFOLIO METRICS (test set)
--------------------------------------------
Total Return     : {pm.get('total_return_pct', 0):.4f}%
Sharpe Ratio     : {pm.get('sharpe_ratio', 0):.4f}
Max Drawdown     : {pm.get('max_drawdown_pct', 0):.4f}%
Win Rate         : {pm.get('win_rate_pct', 0):.2f}%
Profit Factor    : {pm.get('profit_factor', 0):.4f}
Avg Trade Return : {pm.get('avg_trade_pct', 0):.4f}%
Total Trades     : {pm.get('n_trades', 0)}

AVERAGE PER-STOCK METRICS (test set)
--------------------------------------
Sharpe Ratio     : {am.get('sharpe_ratio', 0):.4f}
Total Return     : {am.get('total_return_pct', 0):.4f}%
Win Rate         : {am.get('win_rate_pct', 0):.2f}%
Avg Trade Return : {am.get('avg_trade_pct', 0):.4f}%
Avg Trades/stock : {am.get('n_trades', 0)}

ML-ENHANCED STRATEGY — PORTFOLIO METRICS (test set)
-----------------------------------------------------
Total Return     : {mlm.get('total_return_pct', 0):.4f}%
Sharpe Ratio     : {mlm.get('sharpe_ratio', 0):.4f}
Max Drawdown     : {mlm.get('max_drawdown_pct', 0):.4f}%
Win Rate         : {mlm.get('win_rate_pct', 0):.2f}%
Total Trades     : {mlm.get('n_trades', 0)}

TOP 3 PERFORMING STOCKS (by test Sharpe)
-----------------------------------------
{top3_str}

BOTTOM 3 PERFORMING STOCKS (by test Sharpe)
---------------------------------------------
{bottom3_str}

---

Please provide a structured analysis with EXACTLY these four sections.
Use plain text, no markdown, no asterisks, no bold. Use numbered lists.

SECTION 1 - OVERALL ASSESSMENT (2-3 sentences summarising how this strategy is performing)

SECTION 2 - KEY WEAKNESSES (exactly 3 numbered points, each explaining a specific weakness visible in the numbers)

SECTION 3 - PATTERNS IDENTIFIED (exactly 3 numbered points about patterns you notice across stocks or in the metrics)

SECTION 4 - ACTIONABLE IMPROVEMENTS (exactly 3 numbered points, each a specific, concrete change to parameters, filters, or features that would likely improve performance — be precise, e.g. "increase stop_loss from {stop_loss:.2f}% to X% because...")

Keep each point to 1-2 sentences. Be direct and specific to the numbers shown.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

def _call_groq(prompt: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens":  1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL,
        data    = payload,
        method  = "POST",
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """
    Parse the four-section response into structured lists.
    Tries multiple header styles (SECTION N, numbered, plain headings).
    Falls back to returning everything in 'summary' if parsing fails.
    """
    import re

    sections = {
        "summary":      "",
        "weaknesses":   [],
        "patterns":     [],
        "improvements": [],
    }

    # Build a combined pattern that matches any of these header styles:
    #   "SECTION 1 - OVERALL ASSESSMENT"
    #   "1. OVERALL ASSESSMENT"  /  "1) Overall Assessment"
    #   "OVERALL ASSESSMENT"  (plain heading, uppercase or title-case)
    HEADER_PATTERNS = [
        # explicit SECTION N headers (original format)
        (r"SECTION\s*1[^:\n]*",      r"SECTION\s*2[^:\n]*"),
        (r"SECTION\s*2[^:\n]*",      r"SECTION\s*3[^:\n]*"),
        (r"SECTION\s*3[^:\n]*",      r"SECTION\s*4[^:\n]*"),
        (r"SECTION\s*4[^:\n]*",      r"SECTION\s*5[^:\n]*|$"),
        # numbered list headers "1. Overall Assessment"
        (r"1[\.\)]\s*OVERALL\s+ASSESSMENT[^\n]*",  r"2[\.\)]\s*KEY[^\n]*"),
        (r"2[\.\)]\s*KEY\s+WEAKNESS[^\n]*",         r"3[\.\)]\s*PATTERN[^\n]*"),
        (r"3[\.\)]\s*PATTERN[^\n]*",                r"4[\.\)]\s*(?:ACTIONABLE|IMPROVEMENT)[^\n]*"),
        (r"4[\.\)]\s*(?:ACTIONABLE|IMPROVEMENT)[^\n]*", r"5[\.\)]|$"),
        # plain uppercase headings
        (r"OVERALL\s+ASSESSMENT[^\n]*",  r"KEY\s+WEAKNESS[^\n]*"),
        (r"KEY\s+WEAKNESS[^\n]*",        r"PATTERNS?\s+IDENTIFIED[^\n]*"),
        (r"PATTERNS?\s+IDENTIFIED[^\n]*", r"ACTIONABLE\s+IMPROVEMENTS?[^\n]*"),
        (r"ACTIONABLE\s+IMPROVEMENTS?[^\n]*", r"$"),
    ]

    def try_extract(start_pat, end_pat, src):
        parts = re.split(start_pat, src, flags=re.IGNORECASE)
        if len(parts) < 2:
            return ""
        raw = parts[1]
        cut = re.split(end_pat, raw, flags=re.IGNORECASE)
        return cut[0].strip()

    def bullet_list(raw):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        items = []
        for l in lines:
            cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", l).strip()
            if cleaned:
                items.append(cleaned)
        return items

    # Try each group of four patterns until we get content
    for i in range(0, len(HEADER_PATTERNS), 4):
        p1, p2 = HEADER_PATTERNS[i]
        p3, p4 = HEADER_PATTERNS[i + 1]
        p5, p6 = HEADER_PATTERNS[i + 2]
        p7, p8 = HEADER_PATTERNS[i + 3]

        s = try_extract(p1, p2, text)
        w = try_extract(p3, p4, text)
        p = try_extract(p5, p6, text)
        m = try_extract(p7, p8, text)

        if w or p or m:
            sections["summary"]      = s or text[:400]
            sections["weaknesses"]   = bullet_list(w)[:4]
            sections["patterns"]     = bullet_list(p)[:4]
            sections["improvements"] = bullet_list(m)[:4]
            return sections

    # All patterns failed — put everything in summary so raw display kicks in
    sections["summary"] = text
    return sections
