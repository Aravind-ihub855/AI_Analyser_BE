import os
import httpx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def get_local_events(city: str) -> str:
    """Queries PredictHQ events API for active/upcoming events in a city."""
    api_key = os.getenv("PREDICTHQ_API_KEY")
    if not api_key:
        logger.warning("PREDICTHQ_API_KEY not set in environment.")
        return "Error: PredictHQ API key is not configured."

    url = "https://api.predicthq.com/v1/events/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "q": city,
        "active.gte": current_date,
        "limit": 5,
        "sort": "start"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=12.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results:
                    return f"No active/upcoming events found for: '{city}'."

                formatted_events = []
                for idx, event in enumerate(results, 1):
                    title = event.get("title")
                    category = event.get("category", "General").capitalize()
                    start = event.get("start", "")[:10]
                    venue = event.get("labels", [])
                    venue_str = ", ".join(venue[:3]) if venue else "Local Area"
                    phq_attendance = event.get("phq_attendance", "N/A")
                    
                    formatted_events.append(
                        f"[{idx}] {title} ({category})\n"
                        f"- Date: {start}\n"
                        f"- Target Labels: {venue_str}\n"
                        f"- Projected Attendance: {phq_attendance}"
                    )
                return "\n\n".join(formatted_events)
            else:
                logger.error(f"PredictHQ API returned code {response.status_code} - {response.text}")
                return f"Error: PredictHQ API returned status code {response.status_code}."
    except Exception as e:
        logger.error(f"PredictHQ lookup exception: {str(e)}")
        return f"Error: Exception occurred during PredictHQ lookup: {str(e)}"
