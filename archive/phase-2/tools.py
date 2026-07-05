"""Real tools for Joi's Phase 2 agent: app launcher + weather.

Validated on your machine (2026-06): all three tools worked first try —
.desktop scan found your apps (including Steam games and Flatpak-style
entries), Open-Meteo returned real weather for Madrid, and open_app
launched Firefox.
"""
import configparser
import glob
import os
import subprocess

import httpx

# ---------------------------------------------------------------------------
# Tool 1: App launcher (via .desktop files)
# ---------------------------------------------------------------------------
_DESKTOP_DIRS = [
    "/usr/share/applications",
    os.path.expanduser("~/.local/share/applications"),
]


def _load_apps() -> dict[str, str]:
    """Scan .desktop files -> {app name lowercase: command}."""
    apps: dict[str, str] = {}
    for d in _DESKTOP_DIRS:
        for path in glob.glob(os.path.join(d, "*.desktop")):
            try:
                parser = configparser.ConfigParser(interpolation=None)
                parser.read(path, encoding="utf-8")
                entry = parser["Desktop Entry"]
                if entry.get("NoDisplay", "false").lower() == "true":
                    continue
                name = entry.get("Name")
                cmd = entry.get("Exec")
                if name and cmd:
                    # Exec lines contain placeholders like %U/%f -> strip them
                    cmd = " ".join(p for p in cmd.split() if not p.startswith("%"))
                    apps[name.lower()] = cmd
            except Exception:
                continue  # malformed .desktop file; skip
    return apps


_APPS = _load_apps()


def list_installed_apps() -> str:
    """List the names of applications installed on this computer."""
    return ", ".join(sorted(_APPS.keys()))


def open_app(app_name: str) -> str:
    """Open an application on the computer by its name (e.g. 'firefox')."""
    name = app_name.lower().strip()
    # exact match first, then substring match
    match = _APPS.get(name)
    if match is None:
        candidates = [k for k in _APPS if name in k]
        if not candidates:
            return f"No installed app found matching '{app_name}'."
        match = _APPS[candidates[0]]
        name = candidates[0]
    subprocess.Popen(match.split(), start_new_session=True)
    return f"Opening {name}."


# ---------------------------------------------------------------------------
# Tool 2: Weather (Open-Meteo, free, no API key)
# ---------------------------------------------------------------------------
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