"""Capability: watch anime via ani-cli (installed at /usr/bin/ani-cli).

How it works (verified against the installed script, 2026-07-06):
  - ani-cli's history (~/.local/state/ani-cli/ani-hsts) stores
    "<last_watched_ep>\t<id>\t<title (N episodes)>" and is updated the
    moment an episode starts playing — so "continue" = last + 1.
  - Search results come from the allanime GraphQL API in a deterministic
    order, so we run the same search ourselves to see titles/ids, then tell
    ani-cli which entry to take with -S <index> (never a blind -S 1: for
    "fire force" entry 1 is Season 3 Part 2, and the right one was 6th).
  - English names are resolved by ID INTERSECTION: searching "fire force"
    returns the same id the history stores under "Enen no Shouboutai", so
    the spoken name doesn't need to match the romaji title.
  - mpv is launched detached by ani-cli itself; with no tty its post-play
    menu exits immediately, so the tool returns as soon as playback starts.
"""
import difflib
import json
import os
import re
import subprocess
import time

import httpx

from joi.capabilities import Capability

_HIST = os.path.expanduser("~/.local/state/ani-cli/ani-hsts")
_API = "https://api.allanime.day/api"
_REFR = "https://allmanga.to"
_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
          "Gecko/20100101 Firefox/121.0")
# same query/params the installed ani-cli uses -> same result order as -S
_GQL = ('query( $search: SearchInput $limit: Int $page: Int '
        '$translationType: VaildTranslationTypeEnumType '
        '$countryOrigin: VaildCountryOriginEnumType ) { shows( '
        'search: $search limit: $limit page: $page '
        'translationType: $translationType countryOrigin: $countryOrigin '
        ') { edges { _id name availableEpisodes __typename } }}')


def _search(query: str) -> list[dict]:
    """allanime search, same order ani-cli sees. [{id, title, episodes}]"""
    r = httpx.post(_API, headers={
        "Content-Type": "application/json", "Referer": _REFR,
        "User-Agent": _AGENT,
    }, json={"variables": {
        "search": {"allowAdult": False, "allowUnknown": False, "query": query},
        "limit": 40, "page": 1, "translationType": "sub",
        "countryOrigin": "ALL"}, "query": _GQL}, timeout=20)
    return [{"id": e["_id"], "title": e["name"],
             "episodes": e["availableEpisodes"].get("sub", 0)}
            for e in r.json()["data"]["shows"]["edges"]]


def _history() -> list[dict]:
    """Parsed ani-cli history: [{last_ep, id, title, total}]."""
    entries = []
    try:
        with open(_HIST, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                m = re.match(r"(.*?)\s*\((\d+) episodes\)$", parts[2])
                entries.append({
                    "last_ep": float(parts[0]),
                    "id": parts[1],
                    "title": m.group(1) if m else parts[2],
                    "total": int(m.group(2)) if m else None,
                })
    except OSError:
        pass
    return entries


def _match_history(name: str) -> dict | None:
    """Find a history entry from a spoken name: fuzzy title match first,
    then id-intersection with an allanime search of the spoken name."""
    hist = _history()
    if not hist:
        return None
    name_l = name.lower().strip()
    # direct/fuzzy against stored (romaji) titles
    for h in hist:
        if name_l and (name_l in h["title"].lower()
                       or difflib.SequenceMatcher(
                           None, name_l, h["title"].lower()).ratio() > 0.75):
            return h
    # synonym resolution: search the spoken name, intersect ids with history
    try:
        found_ids = {s["id"] for s in _search(name)}
    except Exception:
        return None
    for h in hist:
        if h["id"] in found_ids:
            return h
    return None


def _mpv_pids() -> set[str]:
    r = subprocess.run(["pgrep", "-x", "mpv"], capture_output=True, text=True)
    return set(r.stdout.split())


def _window_visible(title_fragment: str) -> bool | None:
    """True/False = a window with that title exists (Hyprland); None = can't
    tell (no hyprctl / not in a Hyprland session)."""
    try:
        r = subprocess.run(["hyprctl", "clients", "-j"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        return any(title_fragment.lower() in (c.get("title") or "").lower()
                   for c in json.loads(r.stdout))
    except Exception:
        return None


def _confirm_playback(ep: str, mpv_before: set[str]) -> str:
    """ani-cli found a link and detached mpv — but this provider's links are
    flaky and mpv can die (or hang) with no window (seen live with Jujutsu
    Kaisen 0). mpv only opens a window once video actually decodes, so wait
    for a window titled '... Episode <ep>' before claiming success."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        visible = _window_visible(f"Episode {ep}")
        if visible is None:
            # no Hyprland info — fall back to "did mpv survive a few seconds"
            time.sleep(5)
            return "" if _mpv_pids() - mpv_before else (
                "The player closed right away — that source link is dead at "
                "the moment. Try again, or ask for another variant.")
        if visible:
            return ""  # video is actually on screen
        if not (_mpv_pids() - mpv_before):
            return ("The player closed without showing video — that source "
                    "link is dead at the moment. Try again, or ask for "
                    "another variant.")
        time.sleep(1)
    for pid in _mpv_pids() - mpv_before:  # reap only the mpv WE started
        subprocess.run(["kill", pid], capture_output=True)
    return ("The video never started — the source link seems broken right "
            "now. Try again, or ask for another variant.")


def _run_ani_cli(query: str, index: int, episode: float) -> str:
    """Launch ani-cli non-interactively; returns a speakable status."""
    ep = f"{episode:g}"
    mpv_before = _mpv_pids()
    try:
        r = subprocess.run(
            ["ani-cli", "-S", str(index), "-e", ep, query],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except FileNotFoundError:
        return "ani-cli isn't installed."
    except subprocess.TimeoutExpired:
        return "The video source is taking too long — try again in a moment."
    out = r.stdout + r.stderr
    if "no valid sources" in out:
        return ("The source has no valid video for that episode right now — "
                "try again later.")
    if "not released" in out:
        return "That episode isn't released yet."
    if "Out of range" in out or "No results" in out:
        return "I couldn't find that episode."
    if os.environ.get("ANI_CLI_PLAYER", "mpv") == "mpv":
        return _confirm_playback(ep, mpv_before)
    return ""  # success — caller phrases the announcement


def anime_watch(title: str, episode: int = 0) -> str:
    """Watch an anime by name. Omit `episode` and it CONTINUES where the
    user left off (or starts a new show at episode 1). Pass `episode` ONLY
    when the user says an explicit number ('episode 3 of X'); for 'next
    episode', 'siguiente episodio' or 'keep watching', omit it — the next
    one is worked out from the watch history automatically."""
    entry = _match_history(title)

    if episode <= 0 and entry is not None:
        # continue: history stores the last episode that started playing
        next_ep = entry["last_ep"] + 1
        try:
            results = _search(entry["title"])
            index = next(i for i, s in enumerate(results, 1)
                         if s["id"] == entry["id"])
            total = next(s["episodes"] for s in results if s["id"] == entry["id"])
        except StopIteration:
            return f"I couldn't find {entry['title']} on the source anymore."
        except Exception:
            return "I couldn't reach the anime source right now."
        if next_ep > total:
            return (f"You're all caught up on {entry['title']} — "
                    f"episode {entry['last_ep']:g} of {total} was the last one.")
        err = _run_ani_cli(entry["title"], index, next_ep)
        if err:
            return err.replace("that episode",
                               f"{entry['title']} episode {next_ep:g}")
        return (f"Playing {entry['title']}, episode {next_ep:g} "
                f"of {total}. mpv is opening.")

    episode = max(1, int(episode))
    try:
        results = _search(title)
    except Exception:
        return "I couldn't reach the anime source right now."
    if not results:
        return f"I found no anime matching '{title}'."
    # best candidate by title similarity (shortest name breaks ties: base
    # seasons are shorter than 'X: Season 3 Part 2'), but a history match
    # by id goes first — it resolves English names like 'fire force'
    ranked = sorted(results, key=lambda s: (
        -difflib.SequenceMatcher(None, title.lower(), s["title"].lower()).ratio(),
        len(s["title"])))
    if entry:
        ranked.sort(key=lambda s: s["id"] != entry["id"])
    best = ranked[0]
    if episode > best["episodes"]:
        return f"{best['title']} only has {best['episodes']} episodes."
    # IMPORTANT: ani-cli re-runs the search itself, so -S index and the
    # query string must come from the SAME search — always pass `title`
    # (querying the candidate's own name reorders results and -S would
    # land on a different show; caused Chainsaw Man to play its Recap).
    err = _run_ani_cli(title, results.index(best) + 1, float(episode))
    if not err:
        return (f"Playing {best['title']}, episode {episode} "
                f"of {best['episodes']}. mpv is opening.")
    err = err.replace("that episode",
                      f"{best['title']} episode {episode}")
    # no silent fallback to a different variant (a recap/movie is NOT what
    # the user asked for) — report and name the alternatives instead
    others = [s["title"] for s in ranked[1:4] if s["id"] != best["id"]]
    if others:
        err += (f" Other variants I can try if the user asks: "
                f"{'; '.join(others)}.")
    return err


def anime_progress() -> str:
    """List the anime the user is currently watching and their progress."""
    hist = _history()
    if not hist:
        return "The anime watch history is empty."
    lines = [f"{h['title']}, episode {h['last_ep']:g}"
             + (f" of {h['total']}" if h["total"] else "")
             for h in hist]
    return f"Watching {len(hist)} shows: " + "; ".join(lines) + "."


CAPABILITY = Capability(
    name="anime",
    description=("Watch anime with ani-cli: continue a series where the "
                 "user left off, play the next or a specific episode, or "
                 "list what they're watching."),
    tools=[anime_watch, anime_progress],
)
