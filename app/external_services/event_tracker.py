import os
import httpx
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

async def get_local_events(city: str, days: int = 7) -> str:
    """Queries PredictHQ events API for active/upcoming events in a city within a window of days."""
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
    end_date = (datetime.now() + timedelta(days=max(1, days))).strftime("%Y-%m-%d")
    
    # Query with a larger limit (100) and sort by rank (descending) so we get the major events across the week
    params = {
        **location_param,
        "start.gte": current_date,
        "start.lte": end_date,
        "limit": 100,
        "sort": "rank"
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, headers=headers, params=params, timeout=12.0)
            
            results = []
            if res.status_code == 200:
                results = res.json().get("results", [])
            else:
                logger.warning(f"PredictHQ query returned code {res.status_code}")
                
            if not results:
                return f"No active/upcoming events found for: '{city}' within the next {days} days."
                
            # De-duplicate events by ID to be safe
            seen_ids = set()
            unique_results = []
            for r in results:
                rid = r.get("id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    unique_results.append(r)
            
            # Group events by unique date (YYYY-MM-DD)
            from collections import defaultdict
            events_by_date = defaultdict(list)
            for r in unique_results:
                date_str = r.get("start", "")[:10]
                events_by_date[date_str].append(r)
            
            selected_events = []
            start_dt = datetime.strptime(current_date, "%Y-%m-%d")
            
            # First pass: Pick the single highest rank event for each unique day of the requested window
            for d in range(max(1, days) + 1):
                date_str = (start_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                day_events = events_by_date.get(date_str, [])
                if day_events:
                    # The list unique_results is already sorted by rank descending from the API, 
                    # so the first event in day_events is the highest-ranked one for that day.
                    selected_events.append(day_events[0])
            
            # Second pass: If we have fewer than 8 events, fill up with the next highest-rank events from any day
            if len(selected_events) < 8:
                selected_ids = {e.get("id") for e in selected_events}
                remaining_events = [r for r in unique_results if r.get("id") not in selected_ids]
                for r in remaining_events:
                    if len(selected_events) >= 8:
                        break
                    selected_events.append(r)
                    
            # Sort the final selected events chronologically
            selected_events.sort(key=lambda e: e.get("start", ""))
            
            formatted_events = []
            for idx, event in enumerate(selected_events[:8], 1):
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
