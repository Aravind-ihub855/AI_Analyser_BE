import os
import httpx
import logging
from datetime import datetime, timedelta

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
    
    city_coords = {
        "chicago": "41.8781,-87.6298",
        "houston": "29.7604,-95.3698",
        "los angeles": "34.0522,-118.2437",
        "denver": "39.7392,-104.9903"
    }
    
    city_lower = city.strip().lower()
    location_param = {}
    if city_lower in city_coords:
        location_param = {"within": f"20km@{city_coords[city_lower]}"}
    else:
        location_param = {"q": city}
        
    current_date = datetime.now().strftime("%Y-%m-%d")
    try:
        tomorrow_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        tomorrow_date = current_date
    
    # Query 1: Today's events
    params_today = {
        **location_param,
        "start.gte": current_date,
        "start.lte": current_date,
        "limit": 3,
        "sort": "start"
    }
    
    # Query 2: Future events (tomorrow onwards)
    params_future = {
        **location_param,
        "start.gte": tomorrow_date,
        "limit": 3,
        "sort": "start"
    }

    try:
        async with httpx.AsyncClient() as client:
            res_today = await client.get(url, headers=headers, params=params_today, timeout=12.0)
            res_future = await client.get(url, headers=headers, params=params_future, timeout=12.0)
            
            results = []
            if res_today.status_code == 200:
                results.extend(res_today.json().get("results", []))
            else:
                logger.warning(f"PredictHQ today query returned code {res_today.status_code}")
                
            if res_future.status_code == 200:
                results.extend(res_future.json().get("results", []))
            else:
                logger.warning(f"PredictHQ future query returned code {res_future.status_code}")
                
            if not results:
                return f"No active/upcoming events found for: '{city}'."
                
            # Sort chronologically by start date
            results.sort(key=lambda e: e.get("start", ""))
            
            # De-duplicate events by ID to be safe
            seen_ids = set()
            unique_results = []
            for r in results:
                rid = r.get("id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    unique_results.append(r)
            
            formatted_events = []
            for idx, event in enumerate(unique_results[:5], 1):
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
            
    except Exception as e:
        logger.error(f"PredictHQ lookup exception: {str(e)}")
        return f"Error: Exception occurred during PredictHQ lookup: {str(e)}"
