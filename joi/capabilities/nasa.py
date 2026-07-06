"""Capability: NASA — astronomy picture, near-Earth asteroids, Mars photos,
natural events. All endpoints verified live 2026-07-05 (see CHANGES.md).

Key: NASA_API_KEY in .env (DEMO_KEY fallback works but is limited to
~30 requests/hour). EONET and the NASA Image Library need no key.
"""
import os
import random
import subprocess
from datetime import date, timedelta

import httpx

import joi.config  # noqa: F401  (importing it loads .env)
from joi.capabilities import Capability

_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
_APOD_URL = "https://api.nasa.gov/planetary/apod"
_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
_EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
# The classic Mars Rover Photos API (api.nasa.gov/mars-photos) is dead
# (404 "No such app" at every path, verified 2026-07-05) — we use the
# official NASA Image and Video Library instead.
_IMAGES_URL = "https://images-api.nasa.gov/search"


class _RateLimited(Exception):
    pass


_RATE_LIMIT_MSG = ("NASA's free demo key has hit its hourly request limit. "
                   "Add a NASA_API_KEY to the .env file for a much higher "
                   "limit, or try again in an hour.")


def _nasa_get(url: str, params: dict) -> dict:
    """GET an api.nasa.gov endpoint; raise _RateLimited on 429 so tools can
    answer honestly instead of misreading the error payload."""
    r = httpx.get(url, params=params, timeout=15)
    if r.status_code == 429:
        raise _RateLimited
    r.raise_for_status()
    return r.json()


def _open_in_browser(url: str) -> None:
    subprocess.Popen(["xdg-open", url], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _first_sentences(text: str, n: int = 2) -> str:
    parts = text.replace("\n", " ").split(". ")
    return ". ".join(parts[:n]).strip().rstrip(".") + "."


def nasa_picture_of_the_day() -> str:
    """Get NASA's astronomy picture of the day and show it in the browser."""
    try:
        d = _nasa_get(_APOD_URL, {"api_key": _KEY})
        title = d.get("title", "today's picture")
        gist = _first_sentences(d.get("explanation", ""), 2)
        if d.get("media_type") == "image":
            _open_in_browser(d.get("hdurl") or d["url"])
            return f"Today's astronomy picture is {title}. {gist} I opened it in your browser."
        _open_in_browser(d.get("url", "https://apod.nasa.gov"))
        return (f"Today's astronomy feature is {title}, and it is a video, "
                f"not an image. {gist} I opened it in your browser.")
    except _RateLimited:
        return _RATE_LIMIT_MSG
    except Exception:
        return "I couldn't reach NASA's picture of the day service right now."


def asteroids_near_earth(days: int = 3) -> str:
    """Check asteroids passing near Earth in the next few days (max 7)."""
    try:
        days = max(1, min(int(days), 7))
        start = date.today()
        d = _nasa_get(_NEO_URL, {
            "api_key": _KEY, "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=days)).isoformat(),
        })
        count = d.get("element_count", 0)
        closest, hazardous = None, 0
        for day_objects in d.get("near_earth_objects", {}).values():
            for obj in day_objects:
                if obj.get("is_potentially_hazardous_asteroid"):
                    hazardous += 1
                for ap in obj.get("close_approach_data", []):
                    km = float(ap["miss_distance"]["kilometers"])
                    if closest is None or km < closest[1]:
                        closest = (obj["name"].strip("()"), km,
                                   ap["close_approach_date"])
        if not count or closest is None:
            return f"No near-Earth objects are listed for the next {days} days."
        name, km, when = closest
        msg = (f"{count} objects pass near Earth in the next {days} days. "
               f"The closest is {name} on {when}, "
               f"at about {km / 384400:.1f} times the Moon's distance.")
        if hazardous:
            msg += (f" {hazardous} are in the potentially-hazardous monitoring "
                    "category, which is routine tracking, not an impact warning.")
        return msg
    except _RateLimited:
        return _RATE_LIMIT_MSG
    except Exception:
        return "I couldn't reach NASA's asteroid data right now."


_last_mars_title: str | None = None


def mars_rover_photo(rover: str = "perseverance") -> str:
    """Show a recent Mars rover photo (perseverance or curiosity) in the browser."""
    try:
        d = httpx.get(_IMAGES_URL, params={
            "q": f"{rover} rover", "media_type": "image",
            "year_start": str(date.today().year - 2),
        }, timeout=15).json()
        items = d["collection"]["items"]
        # The library mixes in unrelated event photos, some with bogus
        # FUTURE dates (a 2027-dated briefing, seen live) — keep only items
        # actually titled with the rover's name and dated in the past.
        today = date.today().isoformat()
        items = [it for it in items
                 if rover.lower() in it["data"][0].get("title", "").lower()
                 and it["data"][0].get("date_created", "9999")[:10] <= today]
        if not items:
            return f"I couldn't find recent photos for {rover}."
        # vary the answer: pick among the ~12 most recent, skipping the one
        # shown last time, so asking twice doesn't repeat the same photo
        items.sort(key=lambda it: it["data"][0]["date_created"], reverse=True)
        pool = [it for it in items[:12]
                if it["data"][0]["title"] != _last_mars_title] or items[:1]
        pick = random.choice(pool)
        meta = pick["data"][0]
        globals()["_last_mars_title"] = meta["title"]
        _open_in_browser(pick["links"][0]["href"])
        return (f"I opened a photo from {meta['date_created'][:10]}: "
                f"{meta['title']}.")
    except Exception:
        return "I couldn't reach NASA's image library right now."


def earth_natural_events() -> str:
    """Report natural events happening on Earth now (storms, wildfires, volcanoes)."""
    try:
        d = httpx.get(_EONET_URL, params={"status": "open", "limit": 50},
                      timeout=15).json()
        events = d.get("events", [])
        if not events:
            return "No open natural events are being tracked right now."
        by_cat: dict[str, int] = {}
        for ev in events:
            for cat in ev.get("categories", []):
                by_cat[cat["title"]] = by_cat.get(cat["title"], 0) + 1
        summary = ", ".join(f"{n} {cat.lower()}" for cat, n in
                            sorted(by_cat.items(), key=lambda kv: -kv[1]))
        notable = "; ".join(ev["title"] for ev in events[:2])
        return (f"Earth observatories track {len(events)} open events: "
                f"{summary}. For example: {notable}.")
    except Exception:
        return "I couldn't reach NASA's natural-events service right now."


CAPABILITY = Capability(
    name="nasa",
    description=("NASA space data: astronomy picture of the day, asteroids "
                 "near Earth, Mars rover photos, natural events on Earth."),
    tools=[nasa_picture_of_the_day, asteroids_near_earth,
           mars_rover_photo, earth_natural_events],
)
