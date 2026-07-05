"""Tiny in-process event hooks so capabilities can signal the assistant
without importing it (no circular deps, no framework).

A capability calls emit_game_launched(); whoever registered a callback
(the VoiceAssistant) decides what to do about it.
"""
from typing import Callable

GAME_LAUNCHED: list[Callable[[str], None]] = []


def emit_game_launched(app_name: str) -> None:
    for callback in GAME_LAUNCHED:
        callback(app_name)
