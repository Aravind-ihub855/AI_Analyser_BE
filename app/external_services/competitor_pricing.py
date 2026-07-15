import os
import httpx
import logging
import re

logger = logging.getLogger(__name__)

async def get_competitor_pricing(product_name: str) -> str:
    """Uses Apify (E-commerce / Google Search Scraper actor) to fetch competitor prices from top retailers."""
    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        logger.warning("APIFY_API_KEY not set in environment.")
        return "Error: Apify API key is not configured."

    # Use Google Search Scraper actor to query product pricing
    actor_id = "apify~google-search-scraper"
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items?token={api_key}"
    
    # Target specific competitors using site-operators for maximum accuracy
    targeted_query = f"{product_name} price site:walmart.com OR site:target.com OR site:walgreens.com OR site:cvs.com OR site:kroger.com"
    
    payload = {
        "queries": targeted_query,
        "maxPagesPerQuery": 1,
        "resultsPerPage": 6,
        "countryCode": "us"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=25.0)
            if response.status_code in [200, 201]:
                items = response.json()
                if not items or not isinstance(items, list):
                    return f"No competitor pricing records found for '{product_name}'."
                
                search_data = items[0] if isinstance(items[0], dict) else {}
                organic = search_data.get("organicResults", [])
                
                if not organic:
                    return f"No competitor pricing records found for '{product_name}'."
                
                pricing_table = [
                    "| Competitor | Price | Product Page / Deal |",
                    "| :--- | :--- | :--- |"
                ]
                
                seen_pairs = set()
                for result in organic[:6]:
                    title = result.get("title", "")
                    link = result.get("url", "")
                    snippet = result.get("description") or result.get("snippet", "")
                    
                    # Resolve competitor name
                    competitor = "Other Retailer"
                    link_lower = link.lower()
                    if "walmart.com" in link_lower:
                        competitor = "Walmart"
                    elif "target.com" in link_lower:
                        competitor = "Target"
                    elif "walgreens.com" in link_lower:
                        competitor = "Walgreens"
                    elif "cvs.com" in link_lower:
                        competitor = "CVS Pharmacy"
                    elif "kroger.com" in link_lower:
                        competitor = "Kroger"
                    
                    # Extract price
                    price_match = re.search(r"\$\d+(?:\.\d{2})?", title + " " + snippet)
                    price = price_match.group(0) if price_match else "N/A"
                    
                    # Truncate title for neatness
                    display_title = title[:50] + "..." if len(title) > 50 else title
                    display_title = display_title.replace("|", "-") # avoid markdown syntax breaks
                    
                    pair_key = (competitor, price)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pricing_table.append(f"| {competitor} | **{price}** | [{display_title}]({link}) |")
                
                if len(pricing_table) > 2:
                    return "### Competitor Price Comparison\n\n" + "\n".join(pricing_table)
                else:
                    return "No pricing information could be parsed."
            else:
                logger.error(f"Apify actor failed: status {response.status_code}")
                return f"Apify pricing extraction returned status code {response.status_code}."
    except Exception as e:
        logger.error(f"Apify pricing lookup failed: {str(e)}", exc_info=True)
        return f"Apify pricing actor lookup failed: {str(e)}"
