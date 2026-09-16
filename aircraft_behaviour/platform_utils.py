"""Small platform helpers used by the preserved research scripts."""

from __future__ import annotations

from typing import Iterable, Tuple


def completion_beep(
    pattern: Iterable[Tuple[int, int]] = ((700, 500), (1500, 500), (700, 500))
) -> bool:
    """Play the historical completion tones on Windows.

    Returns True when the tones were played. On non-Windows platforms the
    function quietly returns False so the research script remains portable.
    """
    try:
        import winsound
    except ImportError:
        return False

    for frequency, duration_ms in pattern:
        winsound.Beep(frequency, duration_ms)
    return True
