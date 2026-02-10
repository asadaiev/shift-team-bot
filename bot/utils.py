"""Utility functions."""


def estimate_seconds(msg_count: int, char_count: int) -> int:
    """Estimate time spent typing messages in seconds.
    
    Formula:
    if L <= 100: T = 0.3 * L + 2
    else: T = 0.3 * 100 + 2 + 0.45 * (L - 100)
    where L is total character count.
    """
    L = char_count
    if L <= 100:
        T = 0.3 * L + 2
    else:
        T = 0.3 * 100 + 2 + 0.45 * (L - 100)
    return int(T)


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
