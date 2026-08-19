"""A market brief: one worker per ticker, all of them reading the same tool.

Two things this shows that the other examples do not:

- **A worker tool of your own.** `price_history` is a plain function hitting a
  free endpoint; hand it to `worker_tools` and every worker in the fleet can
  call it. Tools are shared by the whole fleet and carry no task identity, so
  they must be safe to call concurrently — this one only reads.
- **Numbers before prose.** Each per-ticker task is told to pull the series
  first and quote real figures. A model asked about a stock without a tool
  will happily invent a price; a model with one has no reason to.

No API key is needed for the market data (Yahoo's public chart endpoint), only
`ANTHROPIC_API_KEY` for the models.

    uv run python examples/06_market_brief.py NVDA MSFT ASML
"""

import json
import sys
import urllib.request

TICKERS = [t.upper() for t in sys.argv[1:]] or ["NVDA", "MSFT", "ASML"]


def price_history(ticker: str, range: str = "6mo") -> str:
    """Daily closing prices for a ticker.

    Args:
        ticker: Exchange symbol, e.g. "NVDA", "ASML.AS", "^GSPC".
        range: One of 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max.

    Returns:
        A compact JSON object: currency, first/last close, min/max, the
        percentage change over the window, and ~20 evenly spaced samples.
    """
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={range}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "myriapod-example/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception as e:                     # a tool must never kill a worker
        return json.dumps({"ticker": ticker, "error": f"{type(e).__name__}: {e}"})

    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return json.dumps({"ticker": ticker, "error": "unknown ticker"})

    meta = result[0]["meta"]
    stamps = result[0]["timestamp"]
    closes = result[0]["indicators"]["quote"][0]["close"]
    series = [(t, c) for t, c in zip(stamps, closes) if c is not None]
    if not series:
        return json.dumps({"ticker": ticker, "error": "no closing prices"})

    import datetime as dt

    step = max(1, len(series) // 20)
    samples = [
        {"date": dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"), "close": round(c, 2)}
        for t, c in series[::step]
    ]
    first, last = series[0][1], series[-1][1]
    return json.dumps({
        "ticker": ticker,
        "currency": meta.get("currency"),
        "range": range,
        "first_close": round(first, 2),
        "last_close": round(last, 2),
        "change_pct": round((last - first) / first * 100, 2),
        "min_close": round(min(c for _, c in series), 2),
        "max_close": round(max(c for _, c in series), 2),
        "samples": samples,
    })


GOAL = f"""Write a market brief on: {", ".join(TICKERS)}.

Plan one task per ticker. Each one must call `price_history` first and build
its analysis on the figures it returns — six-month trend, drawdown, where the
last close sits in the range — before saying anything qualitative. Then plan a
final task, depending on all of them, that compares the names against each
other and ends with what would have to be true for each thesis to break.

This is analysis, not advice: no price targets, no buy/sell calls. Any figure
that did not come from the tool is a defect."""


def main() -> None:
    from myriapod.adapters.pydantic_ai import build_swarm
    from myriapod.progress import run_with_progress

    swarm = build_swarm(
        planner_model="anthropic:claude-sonnet-5",
        worker_model={
            "standard": "anthropic:claude-haiku-4-5",
            "high": "anthropic:claude-sonnet-5",   # the comparison task
        },
        worker_tools=[price_history],
        max_concurrency=6,
        max_cost=0.50,
        planner_pricing={"input": 3.0, "output": 15.0},
        worker_pricing={
            "standard": {"input": 1.0, "output": 5.0},
            "high": {"input": 3.0, "output": 15.0},
        },
    )

    result = run_with_progress(swarm, GOAL)

    print(result.content or f"no answer: {result.reason}")
    print(f"\n{result.reason} — {result.tasks_done} tasks, "
          f"${result.costs.total_cost:.4f}")


if __name__ == "__main__":
    main()
