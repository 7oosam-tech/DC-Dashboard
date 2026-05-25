import anthropic
import json
import os
import re
from datetime import datetime, timezone

MODEL = "claude-sonnet-4-20250514"
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


def run_claude(prompt):
    """Run Claude with web search in an agentic loop until end_turn."""
    messages = [{"role": "user", "content": prompt}]

    for _ in range(20):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return response

        # Append assistant turn and send back tool results to continue the loop
        messages.append({"role": "assistant", "content": response.content})

        tool_results = [
            {"type": "tool_result", "tool_use_id": block.id, "content": ""}
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        if not tool_results:
            return response  # no tool calls but not end_turn — return anyway

        messages.append({"role": "user", "content": tool_results})

    return response


def extract_json_object(text):
    """Extract the first JSON object found in text."""
    match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*\n)', text)
    if not match:
        match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def fetch_stock(stock):
    """Use Claude + web search to get current data for one stock."""
    prompt = f"""Search the web right now for current stock market data for {stock['name']} (ticker: {stock['ticker']}, listed on {stock['exchange']}, country: {stock['country']}).

Find these exact values today:
1. Current price in {stock['currency']} (local currency)
2. Current price converted to USD (use today's exchange rate if not already USD)
3. Today's percentage change (positive = gain, negative = loss, e.g. 1.25 or -0.80)
4. 52-week high price in USD
5. 52-week low price in USD
6. Analyst consensus price target in USD (if available)
7. Analyst consensus rating text (e.g. "Buy", "Strong Buy", "Hold", "Sell")
8. Number of analysts providing coverage

Search sources like Yahoo Finance, Google Finance, Reuters, Bloomberg, or exchange websites.

Respond with ONLY this JSON object and nothing else:
{{
  "price_local": <number>,
  "price_usd": <number>,
  "change_pct": <number>,
  "high_52w_usd": <number>,
  "low_52w_usd": <number>,
  "analyst_target_usd": <number or null>,
  "analyst_rating": "<string or empty string>",
  "analyst_count": <integer>
}}"""

    try:
        response = run_claude(prompt)
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                data = extract_json_object(text)
                if data and "price_usd" in data:
                    return data
    except Exception as e:
        print(f"  ERROR: {e}")

    return None


def safe_float(val, default=0.0):
    try:
        return round(float(val), 2) if val is not None else default
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    last_updated = now.strftime("%Y-%m-%d %H:%M UTC")

    stocks_data = []

    for stock in STOCKS:
        print(f"\nFetching {stock['name']} ({stock['ticker']})...")
        result = fetch_stock(stock)

        if result:
            entry = {
                "id":                 stock["id"],
                "name":               stock["name"],
                "ticker":             stock["ticker"],
                "exchange":           stock["exchange"],
                "country":            stock["country"],
                "price_local":        safe_float(result.get("price_local")),
                "currency":           stock["currency"],
                "price_usd":          safe_float(result.get("price_usd")),
                "change_pct":         safe_float(result.get("change_pct")),
                "high_52w_usd":       safe_float(result.get("high_52w_usd")),
                "low_52w_usd":        safe_float(result.get("low_52w_usd")),
                "analyst_target_usd": safe_float(result.get("analyst_target_usd")) if result.get("analyst_target_usd") else None,
                "analyst_rating":     str(result.get("analyst_rating") or ""),
                "analyst_count":      safe_int(result.get("analyst_count")),
                "last_search":        today,
            }
            print(f"  price_usd=${entry['price_usd']}  change={entry['change_pct']:+.2f}%  rating={entry['analyst_rating'] or '—'}")
        else:
            entry = {
                "id":                 stock["id"],
                "name":               stock["name"],
                "ticker":             stock["ticker"],
                "exchange":           stock["exchange"],
                "country":            stock["country"],
                "price_local":        0.0,
                "currency":           stock["currency"],
                "price_usd":          0.0,
                "change_pct":         0.0,
                "high_52w_usd":       0.0,
                "low_52w_usd":        0.0,
                "analyst_target_usd": None,
                "analyst_rating":     "",
                "analyst_count":      0,
                "last_search":        today,
            }
            print("  FAILED — using zeros")

        stocks_data.append(entry)

    output = {"last_updated": last_updated, "stocks": stocks_data}

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    succeeded = sum(1 for s in stocks_data if s["price_usd"] > 0)
    print(f"\n✓ data.json saved — {succeeded}/{len(stocks_data)} stocks fetched successfully")
    print(f"  last_updated: {last_updated}")


if __name__ == "__main__":
    main()
