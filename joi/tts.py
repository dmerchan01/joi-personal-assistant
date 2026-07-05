"""Text-to-speech via Piper (piper-tts 1.4.x API, validated on this machine:
PiperVoice.load + voice.synthesize yielding chunks with .audio_int16_bytes
and .sample_rate — synthesize_stream_raw does NOT exist in 1.4.2).

Two layers:
  - TTS: blocking, speak a full string (kept for simple uses and smoke tests).
  - StreamingSpeaker: sentence-pipelined playback. Sentences are queued as the
    LLM produces them and synthesized/played by a background thread, so audio
    starts after sentence 1 instead of after the whole reply.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from piper import PiperVoice


class TTS:
    """Owns the Piper voice; synthesizes and plays text synchronously."""

    def __init__(self, voice_path: str):
        self.voice = PiperVoice.load(voice_path)

    def speak(self, text: str) -> None:
        """Synthesize and play a full string, blocking until playback ends."""
        if not text:
            return
        stream = None
        try:
            stream = self._play_into(text, stream)
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

    def _play_into(self, text: str, stream: sd.OutputStream | None,
                   on_first_audio=None) -> sd.OutputStream | None:
        """Synthesize `text` and write it into `stream`, opening the stream on
        the first chunk (Piper reports the real sample rate per chunk)."""
        for chunk in self.voice.synthesize(text):
            if stream is None:
                stream = sd.OutputStream(
                    samplerate=chunk.sample_rate, channels=1, dtype="int16",
                )
                stream.start()
            if on_first_audio is not None:
                on_first_audio()
                on_first_audio = None
            stream.write(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        return stream


class StreamingSpeaker:
    """One spoken reply, fed sentence by sentence while the LLM still runs.

    Usage per turn:
        speaker = StreamingSpeaker(tts)
        speaker.feed(sentence)      # as each sentence completes (non-blocking)
        await ...                   # keep streaming the LLM meanwhile
        first_audio = speaker.finish()   # blocks until playback done

    first_audio is seconds from StreamingSpeaker creation to the first audio
    samples reaching the output stream (the time-to-first-audio metric).
    """

    _DONE = object()

    def __init__(self, tts: TTS):
        self._tts = tts
        self._queue: queue.Queue = queue.Queue()
        self._t0 = time.perf_counter()
        self._first_audio: float | None = None
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def feed(self, sentence: str) -> None:
        """Queue a sentence for synthesis+playback. Returns immediately."""
        if sentence:
            self._queue.put(sentence)

    def finish(self) -> float | None:
        """Signal end of reply; block until playback ends. Returns
        time-to-first-audio in seconds (None if nothing was spoken)."""
        self._queue.put(self._DONE)
        self._worker.join()
        if self._error is not None:
            raise self._error
        return self._first_audio

    def _mark_first_audio(self) -> None:
        if self._first_audio is None:
            self._first_audio = time.perf_counter() - self._t0

    def _run(self) -> None:
        stream = None
        try:
            while True:
                item = self._queue.get()
                if item is self._DONE:
                    break
                stream = self._tts._play_into(
                    item, stream, on_first_audio=self._mark_first_audio,
                )
        except BaseException as e:  # re-raised in finish()
            self._error = e
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
