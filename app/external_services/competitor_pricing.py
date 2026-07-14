import os
import httpx
import logging

logger = logging.getLogger(__name__)

async def get_competitor_pricing(product_name: str) -> str:
    """Uses Apify (E-commerce / Google Search Scraper actor) to fetch competitor prices."""
    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        logger.warning("APIFY_API_KEY not set in environment.")
        return "Error: Apify API key is not configured."

    # Use Google Search Scraper actor to query product pricing
    actor_id = "apify~google-search-scraper"
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items?token={api_key}"
    
    payload = {
        "queries": f"{product_name} retail price",
        "maxPagesPerQuery": 1,
        "resultsPerPage": 3,
        "countryCode": "us"
    }

    try:
        async with httpx.AsyncClient() as client:
            # Sync execution has a timeout limit
            response = await client.post(url, json=payload, timeout=25.0)
            if response.status_code in [200, 201]:
                items = response.json()
                if not items or not isinstance(items, list):
                    return f"No competitor pricing records found for '{product_name}' on Apify."
                
                # Check for organic results inside the returned items
                pricing_info = []
                search_data = items[0] if isinstance(items[0], dict) else {}
                organic = search_data.get("organicResults", [])
                
                for idx, result in enumerate(organic[:3], 1):
                    title = result.get("title")
                    link = result.get("url")
                    snippet = result.get("description") or result.get("snippet", "")
                    pricing_info.append(
                        f"[{idx}] Competitor: {title}\nURL: {link}\nSnippet: {snippet}"
                    )
                
                if pricing_info:
                    return "\n\n".join(pricing_info)
                
                # Fallback flat list parser
                for idx, item in enumerate(items[:3], 1):
                    title = item.get("title") or item.get("name")
                    price = item.get("price") or item.get("snippet")
                    if title:
                        pricing_info.append(f"[{idx}] {title}: {price}")
                
                return "\n\n".join(pricing_info) if pricing_info else "No pricing info extracted."
            else:
                logger.error(f"Apify actor failed: status {response.status_code}")
                # Fallback pricing response so agent functions smoothly
                return f"Apify pricing extraction returned status {response.status_code}."
    except Exception as e:
        import traceback
        logger.error(f"Apify pricing exception: {str(e)}\n{traceback.format_exc()}")
        return f"Apify pricing actor lookup failed: {str(e)}"
