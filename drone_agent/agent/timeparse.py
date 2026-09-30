"""Turn "after 6 PM", "between 00:10 and 00:20", "before midnight" into a concrete time window.

A question names a wall-clock time ("6 PM") but the stored run has a date and may cross midnight
(23:40 -> 00:30). A clock time therefore has several candidate instants (the same time on each day
the run touches, and the day before). The choice is made relative to the run itself and stated:

* "after X" / "since X": the LATEST X that is not after the run ends ("after 6 PM" on a run that
  starts at 23:40 means the previous evening's 18:00, so it includes the whole run);
* "before X" / "until X": the EARLIEST X that is not before the run starts;
* "between X and Y" / "from X to Y": X as for "after", then the first Y at or after X.

No model is involved and the result is deterministic. Anything it cannot parse returns None, and
the caller answers from unfiltered evidence instead of guessing.
"""

import re
from datetime import datetime, time, timedelta

# A clock time as it appears in a question, without capture groups (it is embedded twice).
_CLOCK = r"(?:\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|midnight|noon|midday)"
# The same, with named groups, for reading one time out of the matched text.
_PARSE = re.compile(
    r"(?:(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?"
    r"|(?P<word>midnight|noon|midday))",
    re.IGNORECASE,
)
_BETWEEN = re.compile(
    rf"\b(?:between|from)\s+(?P<a>{_CLOCK})\s+(?:and|to|until|till)\s+(?P<b>{_CLOCK})",
    re.IGNORECASE,
)
_AFTER = re.compile(rf"\b(?:after|since|later than|past)\s+(?P<t>{_CLOCK})", re.IGNORECASE)
_BEFORE = re.compile(
    rf"\b(?:before|until|till|earlier than|prior to)\s+(?P<t>{_CLOCK})", re.IGNORECASE
)


_AT = re.compile(rf"\b(?:at|around|near)\s+(?P<t>{_CLOCK})", re.IGNORECASE)


def _to_time(text: str) -> time | None:
    m = _PARSE.fullmatch(text.strip())
    if not m:
        return None
    word = (m.group("word") or "").lower()
    if word == "midnight":
        return time(0, 0)
    if word in ("noon", "midday"):
        return time(12, 0)
    hour, minute = int(m.group("h")), int(m.group("m") or 0)
    marker = (m.group("ap") or "").lower().replace(".", "")
    if marker == "pm" and hour < 12:
        hour += 12
    elif marker == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    if not marker and not m.group("m") and hour <= 12 and not text.strip().isdigit():
        return None
    return time(hour, minute)


def _candidates(clock: time, first: datetime, last: datetime) -> list[datetime]:
    day = first.date() - timedelta(days=1)
    out = []
    while day <= last.date() + timedelta(days=1):
        out.append(datetime.combine(day, clock, tzinfo=first.tzinfo))
        day += timedelta(days=1)
    return out


def _after(clock: time, first: datetime, last: datetime) -> datetime:
    return max((c for c in _candidates(clock, first, last) if c <= last), default=first)


def _before(clock: time, first: datetime, last: datetime) -> datetime:
    return min((c for c in _candidates(clock, first, last) if c >= first), default=last)


def parse_time_window(
    question: str, first: datetime, last: datetime
) -> tuple[datetime | None, datetime | None] | None:
    """(start, end) for the time constraint in `question`, or None when it has none we can read.

    `first`/`last` are the run's earliest and latest stored timestamps (timezone-aware).
    """
    q = question.strip()
    between = _BETWEEN.search(q)
    if between:
        a, b = _to_time(between.group("a")), _to_time(between.group("b"))
        if a is None or b is None:
            return None
        start = _after(a, first, last)
        end = min((c for c in _candidates(b, first, last) if c >= start), default=last)
        return start, end
    after = _AFTER.search(q)
    before = _BEFORE.search(q)
    start = end = None
    if after and (t := _to_time(after.group("t"))) is not None:
        start = _after(t, first, last)
    if before and (t := _to_time(before.group("t"))) is not None:
        end = _before(t, first, last)
    if start is None and end is None:
        return None
    return start, end


def parse_moment(question: str, first: datetime, last: datetime) -> datetime | None:
    """The single instant in "at 00:10" / "around midnight", if it falls inside the stored run.

    A clock time has one candidate per day; only one inside [first, last] can be meant, and a time
    outside the run returns None (the caller says so) rather than a guess.
    """
    match = _AT.search(question)
    if not match:
        return None
    clock = _to_time(match.group("t"))
    if clock is None:
        return None
    inside = [c for c in _candidates(clock, first, last) if first <= c <= last]
    return min(inside) if inside else None
