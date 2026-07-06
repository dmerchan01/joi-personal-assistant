"""Capability: filesystem notes under a single root (default ~/Notes,
configurable via JOI_NOTES_ROOT in .env).

Format: one markdown file per folder per day (Notes/ToDo/2026-07-05.md),
each note appended as a "- [HH:MM] text" line — dead simple to read with
any editor and to sync.

SECURITY: every folder name is validated by _safe_folder() and the final
path is realpath-checked to stay inside the root. Voice transcription
errors must never be able to write outside the Notes root; the smoke test
asserts this.
"""
import os
import re
from datetime import datetime

from joi.capabilities import Capability
from joi.config import Config

_ROOT = Config().notes_root


def _safe_folder(folder: str) -> str | None:
    """Return a validated folder name, or None if it tries to escape.
    Only simple names are allowed: letters, digits, spaces, - and _."""
    folder = folder.strip()
    if (not folder or os.path.isabs(folder) or ".." in folder
            or not re.fullmatch(r"[\w][\w \-]*", folder)):
        return None
    path = os.path.realpath(os.path.join(_ROOT, folder))
    if not path.startswith(os.path.realpath(_ROOT) + os.sep):
        return None
    return folder


def add_note(folder: str, content: str) -> str:
    """Save a note into a folder (e.g. ToDo, Projects, Ideas). Creates the
    folder if needed."""
    safe = _safe_folder(folder)
    if safe is None:
        return f"'{folder}' isn't a valid folder name — use a simple name like ToDo."
    if not content.strip():
        return "The note is empty."
    try:
        dirpath = os.path.join(_ROOT, safe)
        os.makedirs(dirpath, exist_ok=True)
        now = datetime.now()
        path = os.path.join(dirpath, now.strftime("%Y-%m-%d") + ".md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- [{now.strftime('%H:%M')}] {content.strip()}\n")
        return f"Noted in {safe}."
    except OSError:
        return "I couldn't write the note to disk."


def read_notes(folder: str) -> str:
    """Read back the most recent notes from a folder."""
    safe = _safe_folder(folder)
    if safe is None:
        return f"'{folder}' isn't a valid folder name."
    dirpath = os.path.join(_ROOT, safe)
    if not os.path.isdir(dirpath):
        return f"There is no {safe} folder yet."
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".md"))
    if not files:
        return f"The {safe} folder is empty."
    notes: list[str] = []
    for name in files:
        with open(os.path.join(dirpath, name), encoding="utf-8") as f:
            notes += [ln.strip("- \n") for ln in f if ln.strip()]
    recent = "; ".join(notes[-3:])
    return f"{safe} has {len(notes)} notes. Most recent: {recent}."


def list_note_folders() -> str:
    """List the note folders that exist."""
    if not os.path.isdir(_ROOT):
        return "You have no notes yet."
    folders = sorted(d for d in os.listdir(_ROOT)
                     if os.path.isdir(os.path.join(_ROOT, d)))
    if not folders:
        return "You have no note folders yet."
    return f"You have {len(folders)} note folders: {', '.join(folders)}."


CAPABILITY = Capability(
    name="notes",
    description=("Personal notes saved on disk by folder (ToDo, Projects, "
                 "Ideas...): add a note, read recent ones, list folders."),
    tools=[add_note, read_notes, list_note_folders],
)
