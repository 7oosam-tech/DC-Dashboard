import anthropic
import json
import os
import re
import time
import yfinance as yf
from datetime import datetime, timezone

MODEL = "claude-sonnet-4-5"
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

STOCKS = [
    {"id": "delta",       "name": "Delta Electronics",               "ticker": "2308.TW",     "exchange": "TWSE",   "country": "Taiwan",      "currency": "TWD"},
    {"id": "liteon",      "name": "Lite-On Technology",              "ticker": "2301.TW",     "exchange": "TWSE",   "country": "Taiwan",      "currency": "TWD"},
    {"id": "ls-electric", "name": "LS Electric",                     "ticker": "010120.KS",   "exchange": "KRX",    "country": "South Korea", "currency": "KRW"},
    {"id": "samsung-sdi", "name": "Samsung SDI",                     "ticker": "006400.KS",   "exchange": "KRX",    "country": "South Korea", "currency": "KRW"},
    {"id": "exide",       "name": "Exide Industries",                "ticker": "EXIDEIND.NS", "exchange": "NSE",    "country": "India",       "currency": "INR"},
    {"id": "amara-raja",  "name": "Amara Raja",                      "ticker": "ARE&M.NS",    "exchange": "NSE",    "country": "India",       "currency": "INR"},
    {"id": "thermax",     "name": "Thermax Limited",                 "ticker": "THERMAX.NS",  "exchange": "NSE",    "country": "India",       "currency": "INR"},
    {"id": "kirloskar",   "name": "Kirloskar Electric",              "ticker": "KECL.NS",     "exchange": "NSE",    "country": "India",       "currency": "INR"},
    {"id": "asetek",      "name": "Asetek",                          "ticker": "ASTK.CO",     "exchange": "CPH",    "country": "Denmark",     "currency": "DKK"},
    {"id": "alfa-laval",  "name": "Alfa Laval",                      "ticker": "ALFA.ST",     "exchange": "STO",    "country": "Sweden",      "currency": "SEK"},
    {"id": "modine",      "name": "Modine Manufacturing",            "ticker": "MOD",         "exchange": "NYSE",   "country": "USA",         "currency": "USD"},
    {"id": "aaon",        "name": "AAON Inc",                        "ticker": "AAON",        "exchange": "NASDAQ", "country": "USA",         "currency": "USD"},
    {"id": "ait",         "name": "Applied Industrial Technologies", "ticker": "AIT",         "exchange": "NYSE",   "country": "USA",         "currency": "USD"},
    {"id": "dnow",        "name": "NOW Inc",                         "ticker": "DNOW",        "exchange": "NYSE",   "country": "USA",         "currency": "USD"},
]

# yfinance FX ticker format: <CCY>USD=X gives units of USD per 1 local unit
FX_PAIRS = {"TWD": "TWDUSD=X", "KRW": "KRWUSD=X", "INR": "INRUSD=X", "DKK": "DKKUSD=X", "SEK": "SEKUSD=X"}


# ── FX rates ──────────────────────────────────────────────────────────────────

def load_fx_rates():
    rates = {"USD": 1.0}
    for ccy, fx_ticker in FX_PAIRS.items():
        try:
            info = yf.Ticker(fx_ticker).fast_info
            rate = info.last_price
            if rate and rate > 0:
                rates[ccy] = rate
                print(f"  FX {ccy}/USD = {rate:.6f}")
            else:
                raise ValueError("no price")
        except Exception as e:
            print(f"  WARNING: FX {ccy}/USD failed ({e}) — USD values will be None for {ccy} stocks")
            rates[ccy] = None
    return rates


# ── yfinance price data ───────────────────────────────────────────────────────

def fetch_price_data(stock, fx_rates):
    ticker = yf.Ticker(stock["ticker"])
    fi = ticker.fast_info

    price_local = fi.last_price or 0.0
    prev_close  = fi.previous_close or 0.0
    high_52w    = fi.year_high or 0.0
    low_52w     = fi.year_low  or 0.0

    change_pct = round((price_local - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    ccy  = stock["currency"]
    rate = fx_rates.get(ccy)

    if ccy == "USD":
        price_usd    = round(price_local, 2)
        high_52w_usd = round(high_52w, 2)
        low_52w_usd  = round(low_52w,  2)
    elif rate:
        price_usd    = round(price_local * rate, 2)
        high_52w_usd = round(high_52w    * rate, 2)
        low_52w_usd  = round(low_52w     * rate, 2)
    else:
        price_usd = high_52w_usd = low_52w_usd = None

    return {
        "price_local":  round(price_local, 2),
        "price_usd":    price_usd,
        "change_pct":   change_pct,
        "high_52w_usd": high_52w_usd,
        "low_52w_usd":  low_52w_usd,
    }


# ── Analyst cache ─────────────────────────────────────────────────────────────

def load_analyst_cache():
    """Read existing data.json and return a dict keyed by stock id."""
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {s["id"]: s for s in data.get("stocks", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def is_cache_fresh(cached, today_str):
    """Return True if the stock was searched within the last 6 days.

    A recent null analyst_target_usd is treated as fresh — the stock has no
    coverage and there is no point retrying Claude every day.
    Only returns False when last_search is missing or older than 6 days.
    """
    if not cached:
        return False
    last_search = cached.get("last_search", "")
    if not last_search:
        return False
    try:
        age = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(last_search, "%Y-%m-%d")).days
        return age <= 6
    except ValueError:
        return False


# ── Claude web search for analyst data ───────────────────────────────────────

def extract_json(text):
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def fetch_analyst_data(stock):
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            timeout=30,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Search for the current analyst consensus price target and rating for "
                    f"{stock['name']} stock ticker {stock['ticker']}. "
                    f"Return ONLY a JSON object with these exact fields: "
                    f"{{\"target_usd\": <number or null>, \"rating\": \"<Strong Buy|Buy|Hold|Sell or empty string>\", "
                    f"\"analyst_count\": <number or 0>}}. Convert target to USD if needed. No other text."
                )
            }]
        )

        print(f"    Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text:
                data = extract_json(block.text)
                if data is not None:
                    return {
                        "analyst_target_usd": data.get("target_usd"),
                        "analyst_rating":     str(data.get("rating") or ""),
                        "analyst_count":      data.get("analyst_count", 0),
                    }
    except Exception as e:
        print(f"    Claude error: {e}")

    return {"analyst_target_usd": None, "analyst_rating": "", "analyst_count": 0}


# ── Main ──────────────────────────────────────────────────────────────────────

def safe_float(v, default=None):
    try:
        return round(float(v), 2) if v is not None else default
    except (TypeError, ValueError):
        return default

def safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    last_updated = now.strftime("%Y-%m-%d %H:%M UTC")

    print("Loading existing analyst cache from data.json...")
    analyst_cache = load_analyst_cache()
    print(f"  Found {len(analyst_cache)} cached entries")

    print("\nLoading FX rates...")
    fx_rates = load_fx_rates()

    stocks_data = []

    for stock in STOCKS:
        print(f"\n{'─'*55}")
        print(f"  {stock['name']} ({stock['ticker']})")

        # 1. Real price data via yfinance
        try:
            prices = fetch_price_data(stock, fx_rates)
            print(f"  yfinance → price={prices['price_local']} {stock['currency']}  "
                  f"USD={prices['price_usd']}  change={prices['change_pct']:+.2f}%  "
                  f"52w [{prices['low_52w_usd']} – {prices['high_52w_usd']}]")
        except Exception as e:
            print(f"  yfinance ERROR: {e}")
            prices = {"price_local": None, "price_usd": None, "change_pct": None,
                      "high_52w_usd": None, "low_52w_usd": None}
        time.sleep(0.5)

        # 2. Analyst data — use cache if fresh (≤6 days), otherwise call Claude
        cached = analyst_cache.get(stock["id"])
        if is_cache_fresh(cached, today):
            age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(cached["last_search"], "%Y-%m-%d")).days
            analyst = {
                "analyst_target_usd": cached.get("analyst_target_usd"),
                "analyst_rating":     cached.get("analyst_rating", ""),
                "analyst_count":      cached.get("analyst_count", 0),
            }
            if analyst["analyst_target_usd"] is None:
                print(f"  analyst → CACHED null (no coverage, age={age}d) — skipping")
            else:
                print(f"  analyst → CACHED (age={age}d)  target={analyst['analyst_target_usd']}  "
                      f"rating={analyst['analyst_rating'] or '—'}  analysts={analyst['analyst_count']}")
        else:
            print(f"  analyst → fetching fresh data via Claude...")
            time.sleep(3)
            analyst = {"analyst_target_usd": None, "analyst_rating": "", "analyst_count": 0}
            for attempt in range(3):
                analyst = fetch_analyst_data(stock)
                if analyst["analyst_target_usd"] is not None:
                    break
                wait = 5 * (attempt + 1)
                print(f"  No data — retrying in {wait}s (attempt {attempt+2}/3)...")
                time.sleep(wait)
            print(f"  analyst → FRESH   target={analyst.get('analyst_target_usd')}  "
                  f"rating={analyst.get('analyst_rating') or '—'}  "
                  f"analysts={analyst.get('analyst_count')}")

        stocks_data.append({
            "id":                 stock["id"],
            "name":               stock["name"],
            "ticker":             stock["ticker"],
            "exchange":           stock["exchange"],
            "country":            stock["country"],
            "price_local":        safe_float(prices["price_local"]),
            "currency":           stock["currency"],
            "price_usd":          safe_float(prices["price_usd"]),
            "change_pct":         safe_float(prices["change_pct"]),
            "high_52w_usd":       safe_float(prices["high_52w_usd"]),
            "low_52w_usd":        safe_float(prices["low_52w_usd"]),
            "analyst_target_usd": safe_float(analyst.get("analyst_target_usd")) if analyst.get("analyst_target_usd") else None,
            "analyst_rating":     str(analyst.get("analyst_rating") or ""),
            "analyst_count":      safe_int(analyst.get("analyst_count")),
            "last_search":        today,
        })

    output = {"last_updated": last_updated, "stocks": stocks_data}

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    succeeded = sum(1 for s in stocks_data if s["price_usd"] is not None and s["price_usd"] > 0)
    print(f"\n{'='*55}")
    print(f"✓ data.json saved — {succeeded}/{len(stocks_data)} stocks with real prices")
    print(f"  last_updated: {last_updated}")


if __name__ == "__main__":
    main()
