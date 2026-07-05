"""Capability: current weather via Open-Meteo (free, no API key).

Validated live 2026-06 for Madrid. This is the only network access in
Joi besides Ollama on localhost.
"""
import httpx

from joi.capabilities import Capability

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Minimal WMO weather-code -> description map (common codes only)
_WMO = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 51: "light drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 95: "thunderstorm",
}


def get_weather(city: str) -> str:
    """Get the current weather for a city (temperature and conditions)."""
    try:
        geo = httpx.get(_GEO_URL, params={"name": city, "count": 1}, timeout=10).json()
        results = geo.get("results")
        if not results:
            return f"Could not find a city named '{city}'."
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        place = results[0]["name"]

        wx = httpx.get(_WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,weather_code",
        }, timeout=10).json()
        cur = wx["current"]
        desc = _WMO.get(cur["weather_code"], "unknown conditions")
        return f"In {place} it is {cur['temperature_2m']} degrees Celsius with {desc}."
    except Exception as e:
        return f"Weather lookup failed: {e}"


CAPABILITY = Capability(
    name="weather",
    description="Current weather (temperature and conditions) for any city.",
    tools=[get_weather],
)
