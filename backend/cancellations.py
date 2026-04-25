"""Persistent cancellation audit log.

When the coordinator clicks "Send Weather Cancellation" we don't have a real
notification channel to send to in this MVP, but we do want the action to be
durable and visible. So we append each cancellation to `data/cancellations.json`
and surface the log in the UI. A future iteration can hook this to email/SMS.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

CANCELLATIONS_FILE = "cancellations.json"

DEFAULT_RECIPIENTS = [
    "Home Coach",
    "Away Coach",
    "Assigned Referee",
    "League Coordinator",
]


@dataclass
class Cancellation:
    match_id: str
    timestamp: str
    reason: str
    notified: List[str]
    forecast_snapshot: dict | None = None


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / CANCELLATIONS_FILE


def load(data_dir: Path) -> List[Cancellation]:
    p = _path(data_dir)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return [Cancellation(**r) for r in rows]


def is_cancelled(data_dir: Path, match_id: str) -> bool:
    return any(c.match_id == match_id for c in load(data_dir))


def add(
    data_dir: Path,
    match_id: str,
    reason: str,
    *,
    notified: List[str] | None = None,
    forecast_snapshot: dict | None = None,
) -> Cancellation:
    rows = load(data_dir)
    rows = [c for c in rows if c.match_id != match_id]  # idempotent
    new = Cancellation(
        match_id=match_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        reason=reason,
        notified=notified or list(DEFAULT_RECIPIENTS),
        forecast_snapshot=forecast_snapshot,
    )
    rows.append(new)
    _path(data_dir).write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return new


def remove(data_dir: Path, match_id: str) -> bool:
    rows = load(data_dir)
    new_rows = [c for c in rows if c.match_id != match_id]
    if len(new_rows) == len(rows):
        return False
    _path(data_dir).write_text(json.dumps([asdict(r) for r in new_rows], indent=2))
    return True


def cancelled_match_ids(data_dir: Path) -> set[str]:
    return {c.match_id for c in load(data_dir)}
