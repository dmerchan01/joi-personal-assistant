"""Capability: tech news via the official Hacker News API
(hacker-news.firebaseio.com/v0 — Firebase-based, free, no key;
verified live 2026-07-05).

Keeps the last fetched list so "open the second one" works.
"""
import subprocess

import httpx

from joi.capabilities import Capability

_BASE = "https://hacker-news.firebaseio.com/v0"

# last tech_news_top result: list of (title, points, url)
_last_stories: list[tuple[str, int, str]] = []


def tech_news_top(n: int = 5) -> str:
    """Get today's top tech news headlines from Hacker News (max 5)."""
    global _last_stories
    try:
        n = max(1, min(int(n), 5))
        ids = httpx.get(f"{_BASE}/topstories.json", timeout=15).json()[:n]
        stories = []
        with httpx.Client(timeout=15) as client:
            for sid in ids:
                item = client.get(f"{_BASE}/item/{sid}.json").json() or {}
                title = item.get("title", "untitled")
                url = item.get("url") or f"https://news.ycombinator.com/item?id={sid}"
                stories.append((title, item.get("score", 0), url))
        _last_stories = stories
        lines = [f"Number {i}: {t}, with {p} points"
                 for i, (t, p, _) in enumerate(stories, 1)]
        return ("Here are the top tech stories. " + ". ".join(lines) +
                ". Say open number one, two and so on to read one.")
    except Exception:
        return "I couldn't reach Hacker News right now."


def tech_news_open(rank: int = 1) -> str:
    """Open tech news story number `rank` from the last list in the browser."""
    if not _last_stories:
        return "I don't have a news list yet — ask for the top tech news first."
    rank = int(rank)
    if not 1 <= rank <= len(_last_stories):
        return f"I only have {len(_last_stories)} stories listed."
    title, _, url = _last_stories[rank - 1]
    subprocess.Popen(["xdg-open", url], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"Opening {title}."


CAPABILITY = Capability(
    name="technews",
    description="Top tech news headlines from Hacker News; can open a story in the browser.",
    tools=[tech_news_top, tech_news_open],
)
