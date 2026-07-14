import os
import httpx
import logging

logger = logging.getLogger(__name__)

async def get_weather(city: str) -> str:
    """Queries OpenWeatherMap API for live weather details of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set in environment.")
        return "Error: OpenWeather API key is not configured."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                main = data.get("main", {})
                weather = data.get("weather", [{}])[0]
                wind = data.get("wind", {})
                
                temp = main.get("temp")
                feels_like = main.get("feels_like")
                humidity = main.get("humidity")
                desc = weather.get("description", "clear sky").capitalize()
                wind_speed = wind.get("speed")

                return (
                    f"Weather in {city.title()}:\n"
                    f"- Condition: {desc}\n"
                    f"- Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
                    f"- Humidity: {humidity}%\n"
                    f"- Wind Speed: {wind_speed} m/s"
                )
            else:
                logger.error(f"OpenWeather error: Status {response.status_code} - {response.text}")
                return f"Error: OpenWeather API returned status code {response.status_code}."
    except Exception as e:
        logger.error(f"Weather lookup exception: {str(e)}")
        return f"Error: Exception occurred during weather lookup: {str(e)}"
