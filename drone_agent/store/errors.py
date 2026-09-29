"""Store exceptions."""

import sqlite3


class StoreError(Exception):
    """Base class for store failures."""


class DuplicateRecordError(StoreError):
    """A record with the same primary/unique key already exists."""


class Fts5UnavailableError(StoreError):
    """This SQLite build lacks FTS5. There is deliberately no fallback search."""


def translate_integrity_error(exc: sqlite3.IntegrityError, what: str) -> StoreError:
    if exc.sqlite_errorcode in (
        sqlite3.SQLITE_CONSTRAINT_UNIQUE,
        sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
    ):
        return DuplicateRecordError(f"{what} already exists ({exc})")
    return StoreError(f"{what} violates a database constraint ({exc})")
