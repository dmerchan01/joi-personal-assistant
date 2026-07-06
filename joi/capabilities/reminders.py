"""Capability: timed reminders (v1: in-memory — reminders die if Joi
restarts; persistence is planned for Phase 4).

When a reminder fires (from a threading.Timer thread) it:
  1. sends a desktop notification via notify-send (mako is running on this
     Hyprland setup; if no daemon answers, the command just no-ops),
  2. speaks through the TTS callable the assistant injects with
     set_speaker(). TTS.speak takes joi.tts.PLAYBACK_LOCK, so a reminder
     never talks over an in-progress reply — it waits for it to finish.
     This was chosen over a checked-between-turns queue because the voice
     loop spends most of its time blocked on input(), where a queue would
     never be drained.
"""
import subprocess
import threading
import time
from typing import Callable

from joi.capabilities import Capability

_speaker: Callable[[str], None] | None = None
_lock = threading.Lock()
_next_id = 1
# id -> (message, fire_at_monotonic, Timer)
_pending: dict[int, tuple[str, float, threading.Timer]] = {}


def set_speaker(speak: Callable[[str], None]) -> None:
    """Called by VoiceAssistant at startup; smoke tests may leave it unset."""
    global _speaker
    _speaker = speak


def _notify(message: str) -> None:
    try:
        subprocess.run(["notify-send", "Joi reminder", message],
                       capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass  # no notification daemon — voice/print still happen


def _fire(rid: int) -> None:
    with _lock:
        entry = _pending.pop(rid, None)
    if entry is None:
        return
    message = entry[0]
    print(f"\n  [reminder] {message}")
    _notify(message)
    if _speaker is not None:
        try:
            _speaker(f"Reminder: {message}")
        except Exception:
            pass  # never let a reminder crash the app


def set_reminder(minutes: float, message: str) -> str:
    """Set a reminder that will be spoken after the given number of minutes."""
    global _next_id
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        return "I need a number of minutes for the reminder."
    if minutes <= 0:
        return "The reminder time must be positive."
    with _lock:
        rid = _next_id
        _next_id += 1
        timer = threading.Timer(minutes * 60, _fire, args=(rid,))
        timer.daemon = True
        _pending[rid] = (message, time.monotonic() + minutes * 60, timer)
        timer.start()
    unit = "minute" if minutes == 1 else "minutes"
    return f"Okay, reminder {rid} set: in {minutes:g} {unit} I'll say {message}."


def list_reminders() -> str:
    """List the reminders that are currently pending."""
    with _lock:
        if not _pending:
            return "There are no pending reminders."
        now = time.monotonic()
        lines = [f"number {rid}, {msg}, in {max(0.0, (at - now) / 60):.0f} minutes"
                 for rid, (msg, at, _) in sorted(_pending.items())]
    return f"{len(lines)} pending: " + "; ".join(lines) + "."


def cancel_reminder(reminder_id: int = 0) -> str:
    """Cancel a reminder by its number, or all reminders if the number is 0."""
    with _lock:
        if not _pending:
            return "There are no reminders to cancel."
        rid = int(reminder_id)
        if rid == 0:
            for _, _, timer in _pending.values():
                timer.cancel()
            n = len(_pending)
            _pending.clear()
            return f"Cancelled all {n} reminders."
        entry = _pending.pop(rid, None)
    if entry is None:
        return f"There is no reminder number {rid}."
    entry[2].cancel()
    return f"Reminder {rid} cancelled."


CAPABILITY = Capability(
    name="reminders",
    description=("Timed reminders: set one for N minutes from now, list "
                 "pending ones, or cancel them. Spoken aloud when they fire."),
    tools=[set_reminder, list_reminders, cancel_reminder],
)
