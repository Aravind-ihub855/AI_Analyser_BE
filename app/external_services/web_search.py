import os
import httpx
import logging

logger = logging.getLogger(__name__)

async def search_web(query: str, max_results: int = 5) -> str:
    """Uses Tavily API to fetch general web search results and summaries."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set in environment.")
        return "Error: Tavily Web Search API key is not configured."

    url = "https://api.tavily.com/search"
    payload = {
      "api_key": api_key,
      "query": query,
      "search_depth": "basic",
      "max_results": max_results
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=15.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results:
                    return f"Web search returned no results for query: '{query}'."

                formatted_results = []
                for idx, item in enumerate(results, 1):
                    formatted_results.append(
                        f"[{idx}] Source: {item.get('title')} ({item.get('url')})\nSnippet: {item.get('content')}"
                    )
                return "\n\n".join(formatted_results)
            else:
                logger.error(f"Tavily API error: Status {response.status_code} - {response.text}")
                return f"Error: Tavily API returned status code {response.status_code}."
    except Exception as e:
        logger.error(f"Tavily search exception: {str(e)}")
        return f"Error: Exception occurred during web search: {str(e)}"
