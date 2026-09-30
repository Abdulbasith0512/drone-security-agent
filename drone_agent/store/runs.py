"""Scenario registry and run/analyst traces (schema v4).

`agent_runs` records what the code observed about a patrol run or an analyst answer (mode,
counters, a step trace). It never stores prompts, credentials or model reasoning.
"""

import json
import sqlite3
from typing import Any

from drone_agent.store.codec import canonical_json


class RunsMixin:
    """Scenario and agent-run methods of `Store` (uses Store's `_conn` and `transaction`)."""

    _conn: sqlite3.Connection

    def upsert_scenario(
        self, name: str, description: str, start_iso: str, spec_sha256: str, expect: dict
    ) -> None:
        with self.transaction():  # type: ignore[attr-defined]
            self._conn.execute(
                "INSERT INTO scenarios (name, description, start, spec_sha256, expect_json)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET description = excluded.description,"
                " start = excluded.start, spec_sha256 = excluded.spec_sha256,"
                " expect_json = excluded.expect_json",
                (name, description, start_iso, spec_sha256, canonical_json(expect)),
            )

    def get_scenario(self, name: str | None = None) -> dict[str, Any] | None:
        """The stored scenario `name`, or the only/first one when `name` is None."""
        if name is None:
            row = self._conn.execute("SELECT * FROM scenarios ORDER BY name LIMIT 1").fetchone()
        else:
            row = self._conn.execute("SELECT * FROM scenarios WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "description": row["description"],
            "start": row["start"],
            "spec_sha256": row["spec_sha256"],
            "expect": json.loads(row["expect_json"]),
        }

    def add_agent_run(
        self,
        kind: str,
        label: str,
        mode: str,
        status: str,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        metrics: dict | None = None,
        trace: list | None = None,
    ) -> int:
        with self.transaction():  # type: ignore[attr-defined]
            cur = self._conn.execute(
                "INSERT INTO agent_runs (kind, label, mode, status, input_text, output_text,"
                " metrics_json, trace_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    label,
                    mode,
                    status,
                    input_text,
                    output_text,
                    canonical_json(metrics or {}),
                    json.dumps(trace or [], sort_keys=True, ensure_ascii=False),
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def update_agent_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        output_text: str | None = None,
        metrics: dict | None = None,
        trace: list | None = None,
    ) -> None:
        with self.transaction():  # type: ignore[attr-defined]
            if status is not None:
                self._conn.execute(
                    "UPDATE agent_runs SET status = ? WHERE run_id = ?", (status, run_id)
                )
            if output_text is not None:
                self._conn.execute(
                    "UPDATE agent_runs SET output_text = ? WHERE run_id = ?", (output_text, run_id)
                )
            if metrics is not None:
                self._conn.execute(
                    "UPDATE agent_runs SET metrics_json = ? WHERE run_id = ?",
                    (canonical_json(metrics), run_id),
                )
            if trace is not None:
                self._conn.execute(
                    "UPDATE agent_runs SET trace_json = ? WHERE run_id = ?",
                    (json.dumps(trace, sort_keys=True, ensure_ascii=False), run_id),
                )

    def list_agent_runs(self, kind: str | None = None) -> list[dict[str, Any]]:
        where, params = ("WHERE kind = ?", [kind]) if kind else ("", [])
        rows = self._conn.execute(
            f"SELECT * FROM agent_runs {where} ORDER BY run_id", params
        ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "kind": r["kind"],
                "label": r["label"],
                "mode": r["mode"],
                "status": r["status"],
                "input_text": r["input_text"],
                "output_text": r["output_text"],
                "metrics": json.loads(r["metrics_json"]),
                "trace": json.loads(r["trace_json"]),
            }
            for r in rows
        ]
