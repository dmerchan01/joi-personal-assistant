"""Capability: launch and list desktop applications (.desktop file scan).

Validated 2026-06: found Steam games and Flatpak entries, launched Firefox,
and the agent routed the Spanish command "Abre Cyberpunk" to open_app.
"""
import configparser
import difflib
import glob
import os
import re
import subprocess

from joi import events
from joi.capabilities import Capability

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


def _tokens(s: str) -> list[str]:
    """'DOOM: The Dark Ages' -> ['doom', 'the', 'dark', 'ages']."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def _find_app(name: str) -> str | None:
    """Match a spoken app name against installed names, most exact first.
    Spoken input is messy: punctuation is lost ('doom dark ages' vs
    'DOOM: The Dark Ages') and Whisper mishears words ('tom raider')."""
    if name in _APPS:
        return name
    candidates = [k for k in _APPS if name in k]
    if not candidates:
        # every spoken word appears in the app name (articles/punctuation-proof)
        want = set(_tokens(name))
        candidates = [k for k in _APPS if want <= set(_tokens(k))]
    if not candidates:
        # fuzzy, for STT mishearings ('tom raider' -> 'tomb raider')
        norm = {" ".join(_tokens(k)): k for k in _APPS}
        close = difflib.get_close_matches(" ".join(_tokens(name)), norm, n=3,
                                          cutoff=0.75)
        candidates = [norm[c] for c in close]
    if not candidates:
        return None
    return max(candidates,
               key=lambda k: difflib.SequenceMatcher(None, name, k).ratio())


def open_app(app_name: str) -> str:
    """Open an application on the computer by its name (e.g. 'firefox').
    Matching is fuzzy — pass the name as the user said it."""
    name = _find_app(app_name.lower().strip())
    if name is None:
        return (f"No installed app found matching '{app_name}'. "
                "Use list_installed_apps to see what is available — or, if "
                "the user wants to watch a show or anime, use anime_watch.")
    cmd = _APPS[name]
    subprocess.Popen(cmd.split(), start_new_session=True)
    if "steam://rungameid" in cmd:
        # Steam games need the whole GPU; let the assistant free the VRAM
        # the LLM is holding (see VoiceAssistant._on_game_launched).
        events.emit_game_launched(name)
    return f"Opening {name}."


CAPABILITY = Capability(
    name="apps",
    description="Open desktop applications and games by name, or list what is installed.",
    tools=[open_app, list_installed_apps],
)
