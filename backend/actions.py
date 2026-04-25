"""Generic action log.

Every "Confirm" button in the dashboard appends a record here. Combined with
`cancellations.json`, this drives the Notifications feed. Keeping it lightweight
JSON on disk so the UI feels stateful across reloads without a database.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

ACTIONS_FILE = "actions.json"


@dataclass
class Action:
    timestamp: str
    kind: str              # "field_reassignment" | "referee_reassignment" | "roster_callup" | "rematch_cancel" | "weekly_defer" | …
    target_match_id: str | None
    summary: str           # human-readable single line
    details: dict | None = None


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / ACTIONS_FILE


def load(data_dir: Path) -> List[Action]:
    p = _path(data_dir)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return [Action(**r) for r in rows]


def add(data_dir: Path, *, kind: str, summary: str,
        target_match_id: str | None = None, details: dict | None = None) -> Action:
    rows = load(data_dir)
    new = Action(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        kind=kind,
        target_match_id=target_match_id,
        summary=summary,
        details=details,
    )
    rows.append(new)
    _path(data_dir).write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return new


def remove_last(data_dir: Path) -> Action | None:
    rows = load(data_dir)
    if not rows:
        return None
    last = rows.pop()
    _path(data_dir).write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return last


def recent(data_dir: Path, limit: int = 5) -> List[Action]:
    rows = load(data_dir)
    return list(reversed(rows))[:limit]
