"""Utility functions."""
from config import Config


def estimate_seconds(msg_count: int, char_count: int) -> int:
    """Estimate time spent typing messages in seconds."""
    typing_seconds = int((char_count / max(Config.TYPING_CHARS_PER_MIN, 1)) * 60)
    overhead_seconds = msg_count * max(Config.SECONDS_OVERHEAD_PER_MSG, 0)
    return typing_seconds + overhead_seconds


def fmt_duration(sec: int) -> str:
    """Format seconds as human-readable duration (Ukrainian)."""
    sec = max(sec, 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h:
        return f"{h}г {m}хв"
    if m:
        return f"{m}хв {s}с"
    return f"{s}с"
