import os
import httpx
import logging

logger = logging.getLogger(__name__)

async def get_market_trend(metric_type: str = "inflation") -> str:
    """Queries Federal Reserve Economic Data (FRED) API for national economic trends."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set in environment.")
        return "Error: FRED API key is not configured."

    # Map request terms to FRED series IDs
    # CPIAUCSL = US CPI, GASREGCOW = Retail Gasoline, UNRATE = Unemployment
    # CPALTT01CAM659N = Canada CPI, CANUR = Canada Unemployment
    series_map = {
        "inflation": "CPIAUCSL",
        "cpi": "CPIAUCSL",
        "fuel": "GASREGCOW",
        "gas_price": "GASREGCOW",
        "unemployment": "UNRATE",
        "canada_inflation": "CPALTT01CAM659N",
        "canada_unemployment": "LRUNTTTTCAQ156S"
    }

    metric = metric_type.lower().strip()
    series_id = series_map.get(metric, "CPIAUCSL")
    metric_label = metric.replace("_", " ").title()

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": 5,
        "sort_order": "desc"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                if not observations:
                    return f"No trend data found for: '{metric_label}'."

                trends = []
                for obs in observations:
                    date = obs.get("date")
                    value = obs.get("value")
                    trends.append(f"- Date: {date} | Value: {value}")
                
                return (
                    f"National Market Trend: {metric_label} ({series_id})\n"
                    f"Latest 5 Monthly Data Points:\n" + "\n".join(trends)
                )
            else:
                logger.error(f"FRED API error: {response.status_code} - {response.text}")
                return f"Error: FRED API returned status code {response.status_code}."
    except Exception as e:
        logger.error(f"FRED lookup exception: {str(e)}")
        return f"Error: Exception occurred during FRED lookup: {str(e)}"
