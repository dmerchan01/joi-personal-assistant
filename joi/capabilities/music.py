"""Capability: music control.

Stage A (working): control whatever MPRIS player is running via playerctl
(installed, 2.4.1; `-p NAME` flag verified with --help). Spotify is tried
first, then any other player.

Stage B (scaffold only): playing a SPECIFIC song/artist needs the Spotify
Web API — a developer app, OAuth, and (for playback endpoints) Premium.
See README "Spotify search (Stage B)" for the setup steps. The stubs below
answer honestly until credentials are configured; do not invent a flow.
"""
import shutil
import subprocess

from joi.capabilities import Capability


def _playerctl(*args: str) -> tuple[bool, str]:
    """Run playerctl, preferring spotify, falling back to any player.
    Returns (ok, output)."""
    if shutil.which("playerctl") is None:
        return False, ("playerctl is not installed. Install it with "
                       "pacman: sudo pacman -S playerctl.")
    for selector in (["-p", "spotify"], []):
        try:
            r = subprocess.run(["playerctl", *selector, *args],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return True, r.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            break
    return False, "No music player seems to be running."


def music_pause() -> str:
    """Pause the music that is currently playing."""
    ok, out = _playerctl("pause")
    return "Music paused." if ok else out


def music_resume() -> str:
    """Resume or start playing music."""
    ok, out = _playerctl("play")
    return "Music playing." if ok else out


def music_next() -> str:
    """Skip to the next song."""
    ok, out = _playerctl("next")
    return "Skipped to the next track." if ok else out


def music_previous() -> str:
    """Go back to the previous song."""
    ok, out = _playerctl("previous")
    return "Went back to the previous track." if ok else out


def music_now_playing() -> str:
    """Check which song and artist is currently playing on the music player.
    Use whenever the user asks what song is playing or what this song is."""
    ok, out = _playerctl("metadata", "--format", "{{artist}} - {{title}}")
    if not ok:
        return out
    return f"Now playing: {out}." if out.strip(" -") else "Nothing seems to be playing."


def music_play_song(query: str) -> str:
    """Play a specific song, artist or playlist by name (needs Spotify setup)."""
    # TODO(Stage B): Spotify Web API search + start playback. Requires
    # SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env and an OAuth flow —
    # see README. Not implemented until real credentials exist to test with.
    return ("Playing a specific song isn't configured yet — it needs the "
            "Spotify setup described in the README. I can pause, resume "
            "and skip what's already playing.")


CAPABILITY = Capability(
    name="music",
    description=("Control the music player: pause, resume, next, previous, "
                 "what's playing. Playing a specific song needs Spotify setup."),
    tools=[music_pause, music_resume, music_next, music_previous,
           music_now_playing, music_play_song],
)
