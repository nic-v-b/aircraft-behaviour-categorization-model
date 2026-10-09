"""Small time-conversion helpers shared by the research scripts."""

import datetime


_INTERVALS = (
    ("weeks", 604800),
    ("days", 86400),
    ("hours", 3600),
    ("minutes", 60),
    ("seconds", 1),
)


def unix_to_local(unix_time):
    """Convert a Unix timestamp to a naive local datetime."""
    return datetime.datetime.fromtimestamp(unix_time)


def unix_to_utc(unix_time):
    """Convert a Unix timestamp to a naive UTC datetime."""
    return datetime.datetime.utcfromtimestamp(unix_time)


def display_time(seconds, granularity=2):
    """Format a duration using the requested number of non-zero units."""
    result = []
    for name, count in _INTERVALS:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip("s")
            result.append(f"{value} {name}")
    return ", ".join(result[:granularity])
