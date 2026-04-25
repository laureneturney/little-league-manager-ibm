"""Schedule mutations.

When a coordinator clicks **Confirm** on an AI Suggestion, the alert's
structured payload is translated into a list of low-level operations
(`set_field`, `set_referee`, `bump_attendance`, `move_match`, `remove_match`)
which are applied to the live `LeagueData.matches` DataFrame *and* appended to
`data/mutations.json`. On startup the persisted ops are replayed so changes
survive Streamlit restarts.

The diagnostics pipeline reads from the same live DataFrame, so applied
suggestions immediately drop their associated alerts and counts.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .data_loader import LeagueData, _datetime_at, _parse_time

MUTATIONS_FILE = "mutations.json"
MIN_PLAYERS = 9


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / MUTATIONS_FILE


def load(data_dir: Path) -> List[Dict[str, Any]]:
    p = _path(data_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def _save(data_dir: Path, rows: List[Dict[str, Any]]) -> None:
    _path(data_dir).write_text(json.dumps(rows, indent=2))


def append_record(data_dir: Path, record: Dict[str, Any]) -> None:
    rows = load(data_dir)
    rows.append(record)
    _save(data_dir, rows)


def clear_all(data_dir: Path) -> None:
    p = _path(data_dir)
    if p.exists():
        p.unlink()


def replay(data: LeagueData, data_dir: Path) -> int:
    """Apply all persisted mutations to a freshly-loaded LeagueData."""
    rows = load(data_dir)
    for r in rows:
        for op in r.get("ops", []):
            apply_op(data, op)
    return len(rows)


# ----------------------------------------------------------------------
# Public: translate a high-level alert/suggestion to ops + apply + persist
# ----------------------------------------------------------------------
def apply_suggestion(
    data: LeagueData, data_dir: Path, alert: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply a suggestion to the live data. Returns a record with the ops
    that were applied. Persists the record so it can be replayed later."""
    sug = alert.get("suggestion") or {}
    kind = sug.get("type") or alert.get("category") or "unknown"
    ops: List[Dict[str, Any]] = []

    if kind == "field_reassignment":
        match_id = sug.get("match_id")
        new_field = sug.get("new_field_id")
        if match_id and new_field:
            ops.append({"op": "set_field", "match_id": match_id, "field_id": new_field})
        elif match_id and sug.get("fallback_slot"):
            slot = sug["fallback_slot"]
            ops.append({
                "op": "move_match", "match_id": match_id,
                "date": slot["date"], "time_start": slot["time_start"],
                "time_end": slot["time_end"],
            })

    elif kind == "referee_reassignment":
        match_id = sug.get("match_id")
        new_ref = sug.get("new_ref_id")
        if match_id and new_ref:
            ops.append({"op": "set_referee", "match_id": match_id, "ref_id": new_ref})

    elif kind == "roster_callup":
        match_id = sug.get("match_id")
        team_id = sug.get("team_id")
        if match_id and team_id:
            ops.append({
                "op": "bump_attendance", "match_id": match_id,
                "team_id": team_id, "to": MIN_PLAYERS,
            })

    elif kind == "weather_reschedule":
        # Per-day bundle: reschedule every match on that day.
        for entry in sug.get("match_alternatives", []):
            mid, alt = entry.get("match_id"), entry.get("alternative")
            if mid and alt:
                ops.append({
                    "op": "move_match", "match_id": mid,
                    "date": alt["date"], "time_start": alt["time_start"],
                    "time_end": alt["time_end"],
                    "field_id": alt.get("field_id"), "ref_id": alt.get("ref_id"),
                })

    elif kind == "cancel_rematch":
        if sug.get("match_id"):
            ops.append({"op": "remove_match", "match_id": sug["match_id"]})

    elif kind == "defer_match":
        for mid in sug.get("match_ids", []):
            ops.append({"op": "remove_match", "match_id": mid})

    # Apply now to the live data
    for op in ops:
        apply_op(data, op)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "alert_title": alert.get("title"),
        "ops": ops,
    }
    if ops:
        append_record(data_dir, record)
    return record


# ----------------------------------------------------------------------
# Low-level ops
# ----------------------------------------------------------------------
def apply_op(data: LeagueData, op: Dict[str, Any]) -> None:
    op_type = op["op"]
    if op_type == "set_field":
        _set_field(data, op["match_id"], op["field_id"])
    elif op_type == "set_referee":
        _set_referee(data, op["match_id"], op["ref_id"])
    elif op_type == "bump_attendance":
        _bump_attendance(data, op["match_id"], op["team_id"], int(op["to"]))
    elif op_type == "move_match":
        _move_match(
            data, op["match_id"],
            date=op.get("date"),
            time_start=op.get("time_start"),
            time_end=op.get("time_end"),
            field_id=op.get("field_id"),
            ref_id=op.get("ref_id"),
        )
    elif op_type == "remove_match":
        _remove_match(data, op["match_id"])


def _mask(data: LeagueData, match_id: str):
    return data.matches["MatchID"] == match_id


def _set_field(data: LeagueData, match_id: str, new_field_id: str) -> None:
    m = _mask(data, match_id)
    if m.any():
        data.matches.loc[m, "FieldID"] = new_field_id


def _set_referee(data: LeagueData, match_id: str, new_ref_id: str) -> None:
    m = _mask(data, match_id)
    if m.any():
        data.matches.loc[m, "RefID"] = new_ref_id


def _bump_attendance(data: LeagueData, match_id: str, team_id: str, to: int) -> None:
    m = _mask(data, match_id)
    if not m.any():
        return
    row = data.matches[m].iloc[0]
    col = "HomePresent" if row["HomeTeamID"] == team_id else "AwayPresent"
    if int(data.matches.loc[m, col].iloc[0]) < to:
        data.matches.loc[m, col] = to


def _move_match(
    data: LeagueData, match_id: str, *,
    date: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    field_id: str | None = None,
    ref_id: str | None = None,
) -> None:
    m = _mask(data, match_id)
    if not m.any():
        return
    if date is not None:
        data.matches.loc[m, "Date"] = date
    if time_start is not None:
        data.matches.loc[m, "TimeStart"] = time_start
    if time_end is not None:
        data.matches.loc[m, "TimeEnd"] = time_end
    if field_id is not None:
        data.matches.loc[m, "FieldID"] = field_id
    if ref_id is not None:
        data.matches.loc[m, "RefID"] = ref_id

    # Recompute derived columns
    cur_date = data.matches.loc[m, "Date"].iloc[0]
    cur_ts = data.matches.loc[m, "TimeStart"].iloc[0]
    cur_te = data.matches.loc[m, "TimeEnd"].iloc[0]
    data.matches.loc[m, "StartDT"] = _datetime_at(cur_date, cur_ts)
    data.matches.loc[m, "EndDT"] = _datetime_at(cur_date, cur_te)
    data.matches.loc[m, "Time24"] = _parse_time(cur_ts).strftime("%H:%M")
    data.matches.loc[m, "TimeSlot"] = f"{cur_ts} - {cur_te}"


def _remove_match(data: LeagueData, match_id: str) -> None:
    data.matches = data.matches[data.matches["MatchID"] != match_id].reset_index(drop=True)
