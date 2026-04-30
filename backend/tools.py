"""Agent tools.

Every tool here is a pure function that takes a `LeagueData` and structured
arguments, and returns JSON-serializable data. The agent (LLM- or rule-driven)
composes these to produce actionable, structured suggestions for the UI.

Each suggestion always includes a SPECIFIC resolution — Field #, Time, and/or
Referee name — so the UI can render an "Apply" button that ingests it directly.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import LeagueData, _parse_time
from .weather import MockWeatherProvider, WeatherProvider

MIN_PLAYERS = 9
HOME_AWAY_TOLERANCE = 2    # |home - away| should be ≤ this  (legacy; not in default report)


# ------------------------------------------------------------------
# Lookups
# ------------------------------------------------------------------
def get_upcoming_matches(data: LeagueData, today: str, days: int = 14) -> List[Dict[str, Any]]:
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    end_dt = today_dt + timedelta(days=days)
    df = data.matches[(data.matches["StartDT"] >= today_dt) & (data.matches["StartDT"] < end_dt)]
    df = df.sort_values("StartDT")
    return [_match_to_dict(data, r) for _, r in df.iterrows()]


def get_match_details(data: LeagueData, match_id: str) -> Optional[Dict[str, Any]]:
    rows = data.matches[data.matches["MatchID"] == match_id]
    if rows.empty:
        return None
    return _match_to_dict(data, rows.iloc[0])


def get_team_roster(data: LeagueData, team_id: str) -> Dict[str, Any]:
    players = data.roster[data.roster["TeamID"] == team_id]["PlayerName"].tolist()
    return {
        "team_id": team_id,
        "team_name": data.team_name(team_id),
        "starters": players[:MIN_PLAYERS],
        "standby": players[MIN_PLAYERS:],
        "total": len(players),
    }


def get_standby_players(data: LeagueData, team_id: str) -> List[str]:
    return get_team_roster(data, team_id)["standby"]


def resolve_team_id(data: LeagueData, query: str) -> Optional[str]:
    """Best-effort team lookup from name or ID. Case-insensitive."""
    if not query:
        return None
    q = str(query).strip().lower()
    for tid, name in data.teams.items():
        if q == tid.lower() or q == name.lower():
            return tid
    for tid, name in data.teams.items():
        if q in name.lower() or name.lower() in q:
            return tid
    return None


def get_team_matches(
    data: LeagueData, team_id: str,
    today: Optional[str] = None, limit: int = 5,
) -> List[Dict[str, Any]]:
    """Matches for a team, future-only when `today` is given. Up to `limit` rows."""
    tid = resolve_team_id(data, team_id) or team_id
    df = data.matches[
        (data.matches["HomeTeamID"] == tid) | (data.matches["AwayTeamID"] == tid)
    ]
    if today:
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        df = df[df["StartDT"] >= today_dt]
    df = df.sort_values("StartDT").head(int(limit))
    return [_match_to_dict(data, r) for _, r in df.iterrows()]


def get_standings(data: LeagueData, today: str, top: int = 12) -> List[Dict[str, Any]]:
    """League standings (W/L/T/PTS) computed from matches whose date < today."""
    from .standings import standings_table  # local import to avoid cycles
    return standings_table(data, today)[: int(top)]


# ------------------------------------------------------------------
# Validators (return structured Alert dicts)
# ------------------------------------------------------------------
def check_field_conflicts(data: LeagueData, today: str) -> List[Dict[str, Any]]:
    """Two matches booked on the same field at the same date+time."""
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    upcoming = data.matches[data.matches["StartDT"] >= today_dt]
    alerts: List[Dict[str, Any]] = []
    grouped = upcoming.groupby(["Date", "TimeStart", "FieldID"])
    for (date, time_start, field_id), grp in grouped:
        if len(grp) < 2:
            continue
        match_ids = grp["MatchID"].tolist()
        free_field = _first_free_field(data, date, time_start, exclude_field=field_id, ignore_match_ids=match_ids)
        free_slot = _first_free_slot_for_field(data, date, field_id, exclude_match_ids=match_ids)
        suggestion: Dict[str, Any] = {
            "type": "field_reassignment",
            "match_id": match_ids[1],
            "new_field_id": free_field,
            "new_field_name": data.field_name(free_field) if free_field else None,
            "fallback_slot": free_slot,
        }
        text = (
            f"Move match {match_ids[1]} to {data.field_name(free_field)} "
            f"({field_id} → {free_field}) at {time_start}."
            if free_field else
            f"No alternate field free at {time_start} on {date}; "
            f"shift match {match_ids[1]} to {free_slot['time_start']}–{free_slot['time_end']} on {field_id}."
            if free_slot else
            f"No conflict-free option found — escalate to coordinator."
        )
        alerts.append({
            "severity": "critical",
            "category": "field_conflict",
            "title": f"Field {field_id} double-booked on {date} {time_start}",
            "description": (
                f"{len(match_ids)} matches scheduled on {data.field_name(field_id)} "
                f"at the same slot: {', '.join(match_ids)}."
            ),
            "match_ids": match_ids,
            "date": date,
            "time": time_start,
            "suggestion": suggestion,
            "suggestion_text": text,
        })
    return alerts


def check_referee_conflicts(data: LeagueData, today: str) -> List[Dict[str, Any]]:
    """One referee assigned to two matches in the same date+time slot."""
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    upcoming = data.matches[data.matches["StartDT"] >= today_dt]
    alerts: List[Dict[str, Any]] = []
    grouped = upcoming.groupby(["Date", "TimeStart", "RefID"])
    for (date, time_start, ref_id), grp in grouped:
        if len(grp) < 2:
            continue
        match_ids = grp["MatchID"].tolist()
        free_ref = _first_free_referee(data, date, time_start, exclude_ref=ref_id, ignore_match_ids=match_ids)
        suggestion = {
            "type": "referee_reassignment",
            "match_id": match_ids[1],
            "new_ref_id": free_ref,
            "new_ref_name": data.referee_name(free_ref) if free_ref else None,
        }
        text = (
            f"Reassign match {match_ids[1]} to referee {data.referee_name(free_ref)} ({free_ref})."
            if free_ref else
            f"No alternate referee free at {time_start} on {date} — escalate."
        )
        alerts.append({
            "severity": "critical",
            "category": "referee_conflict",
            "title": f"Referee {data.referee_name(ref_id)} double-booked on {date} {time_start}",
            "description": (
                f"Ref {ref_id} ({data.referee_name(ref_id)}) is assigned to "
                f"{len(match_ids)} simultaneous matches: {', '.join(match_ids)}."
            ),
            "match_ids": match_ids,
            "date": date,
            "time": time_start,
            "suggestion": suggestion,
            "suggestion_text": text,
        })
    return alerts


def check_roster_shortage(data: LeagueData, today: str) -> List[Dict[str, Any]]:
    """Alert only when a team has FEWER than MIN_PLAYERS players for an
    upcoming match (i.e., would forfeit). 'Tight roster' (exactly 9) is
    intentionally NOT alerted."""
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    upcoming = data.matches[data.matches["StartDT"] >= today_dt]
    alerts: List[Dict[str, Any]] = []
    for _, m in upcoming.iterrows():
        for side, count_col, team_col in (
            ("home", "HomePresent", "HomeTeamID"),
            ("away", "AwayPresent", "AwayTeamID"),
        ):
            present = int(m[count_col])
            if present >= MIN_PLAYERS:
                continue
            team_id = m[team_col]
            opponent_id = m["AwayTeamID"] if side == "home" else m["HomeTeamID"]
            standby = get_standby_players(data, team_id)
            needed = MIN_PLAYERS - present
            call_ups = standby[:needed]
            severity = "warning" if call_ups else "critical"
            title = f"Roster Shortage: {data.team_name(team_id)} ({present}/{MIN_PLAYERS}) on {m['Date']}"
            description = (
                f"{data.team_name(team_id)} has only {present} players for match "
                f"{m['MatchID']} vs {data.team_name(opponent_id)} on {m['Date']} "
                f"{m['TimeStart']}. League minimum is {MIN_PLAYERS}."
            )
            if call_ups:
                text = (
                    f"Call up {len(call_ups)} substitute(s) from standby for "
                    f"{data.team_name(team_id)}: {', '.join(call_ups)}."
                )
            else:
                text = (
                    f"{data.team_name(team_id)} has no standby players left — "
                    f"team will forfeit unless coordinator intervenes."
                )
            alerts.append({
                "severity": severity,
                "category": "roster_shortage",
                "title": title,
                "description": description,
                "match_ids": [m["MatchID"]],
                "date": m["Date"],
                "time": m["TimeStart"],
                "suggestion": {
                    "type": "roster_callup",
                    "match_id": m["MatchID"],
                    "team_id": team_id,
                    "team_name": data.team_name(team_id),
                    "needed": needed,
                    "call_ups": call_ups,
                },
                "suggestion_text": text,
            })
    return alerts


def check_rematch_violations(
    data: LeagueData, today: str, max_count: int | None = None
) -> List[Dict[str, Any]]:
    """A pair of teams scheduled to play more than `max_count` times in total.

    `max_count is None` (or <= 0) means the rule is disabled — no alerts.
    """
    if max_count is None or max_count <= 0:
        return []
    counts: Dict[tuple, list] = defaultdict(list)
    for _, m in data.matches.iterrows():
        pair = tuple(sorted([m["HomeTeamID"], m["AwayTeamID"]]))
        counts[pair].append((m["MatchID"], m["Date"]))
    alerts: List[Dict[str, Any]] = []
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    for pair, instances in counts.items():
        if len(instances) <= max_count:
            continue
        future = [i for i in instances if datetime.strptime(i[1], "%Y-%m-%d") >= today_dt]
        if not future:
            continue  # already played, can't fix
        target = sorted(future, key=lambda x: x[1])[-1]
        alerts.append({
            "severity": "warning",
            "category": "rematch_violation",
            "title": f"Teams {pair[0]} vs {pair[1]} scheduled {len(instances)} times (limit {max_count})",
            "description": (
                f"{data.team_name(pair[0])} and {data.team_name(pair[1])} are booked "
                f"to play {len(instances)} times. Coordinator-set cap is {max_count}."
            ),
            "match_ids": [i[0] for i in instances],
            "date": target[1],
            "time": None,
            "suggestion": {
                "type": "cancel_rematch",
                "match_id": target[0],
                "pair": list(pair),
            },
            "suggestion_text": (
                f"Cancel match {target[0]} on {target[1]} between "
                f"{data.team_name(pair[0])} and {data.team_name(pair[1])}."
            ),
        })
    return alerts


def check_weekly_game_limit(
    data: LeagueData, today: str, max_count: int | None = None
) -> List[Dict[str, Any]]:
    """A team appearing in `max_count` or more matches in a Mon-Sun week.

    `max_count is None` (or <= 0) means the rule is disabled — no alerts.
    """
    if max_count is None or max_count <= 0:
        return []
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    upcoming = data.matches[data.matches["StartDT"] >= today_dt].copy()
    upcoming["Date"] = pd.to_datetime(upcoming["Date"])
    upcoming["WeekStart"] = upcoming["Date"] - pd.to_timedelta(upcoming["Date"].dt.weekday, unit="d")
    alerts: List[Dict[str, Any]] = []

    rows = []
    for _, m in upcoming.iterrows():
        rows.append({"team_id": m["HomeTeamID"], "match_id": m["MatchID"], "date": m["Date"], "week": m["WeekStart"]})
        rows.append({"team_id": m["AwayTeamID"], "match_id": m["MatchID"], "date": m["Date"], "week": m["WeekStart"]})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return alerts
    grouped = long_df.groupby(["team_id", "week"])
    for (team_id, week), grp in grouped:
        if len(grp) < max_count:
            continue
        sorted_grp = grp.sort_values("date")
        # Defer everything beyond max_count - 1 (i.e., trim down so the team
        # is at most max_count - 1 games for that week).
        keep = max(0, max_count - 1)
        excess_matches = sorted_grp.iloc[keep:]["match_id"].tolist()
        alerts.append({
            "severity": "warning",
            "category": "weekly_game_limit",
            "title": (
                f"{data.team_name(team_id)} has {len(grp)} games in week of "
                f"{week.date().isoformat()} (cap ≥ {max_count})"
            ),
            "description": (
                f"{data.team_name(team_id)} ({team_id}) is at or above the weekly cap. "
                f"Matches: {', '.join(grp['match_id'].tolist())}."
            ),
            "match_ids": grp["match_id"].tolist(),
            "date": week.date().isoformat(),
            "time": None,
            "suggestion": {
                "type": "defer_match",
                "match_ids": excess_matches,
                "team_id": team_id,
            },
            "suggestion_text": (
                f"Defer match(es) {', '.join(excess_matches)} for {data.team_name(team_id)} "
                f"to bring the week back under the cap."
            ),
        })
    return alerts


def check_home_away_balance(data: LeagueData) -> List[Dict[str, Any]]:
    """Teams whose home/away spread is wider than HOME_AWAY_TOLERANCE."""
    home_counts = Counter(data.matches["HomeTeamID"])
    away_counts = Counter(data.matches["AwayTeamID"])
    alerts: List[Dict[str, Any]] = []
    for team_id in data.teams:
        h, a = home_counts[team_id], away_counts[team_id]
        diff = h - a
        if abs(diff) <= HOME_AWAY_TOLERANCE:
            continue
        alerts.append({
            "severity": "info",
            "category": "home_away_imbalance",
            "title": f"{data.team_name(team_id)} home/away imbalance: {h}H / {a}A",
            "description": (
                f"{data.team_name(team_id)} ({team_id}) has {h} home games and {a} away games "
                f"(spread {diff:+d}, tolerance ±{HOME_AWAY_TOLERANCE})."
            ),
            "match_ids": [],
            "date": None,
            "time": None,
            "suggestion": {
                "type": "swap_home_away",
                "team_id": team_id,
                "direction": "more_away" if diff > 0 else "more_home",
            },
            "suggestion_text": (
                f"Flip home/away on a future {data.team_name(team_id)} match to balance the schedule."
            ),
        })
    return alerts


# ------------------------------------------------------------------
# Weather-driven rescheduling
# ------------------------------------------------------------------
def check_weather_alerts(
    data: LeagueData,
    today: str,
    weather_provider: WeatherProvider,
    location: Dict[str, Any],
    days: int = 14,
) -> List[Dict[str, Any]]:
    """One alert per severe-forecast day, bundling every match scheduled
    that day. The suggestion carries an `match_alternatives` list so the
    UI's Confirm button can reschedule all of them in one click."""
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    end_dt = today_dt + timedelta(days=days)
    upcoming = data.matches[
        (data.matches["StartDT"] >= today_dt) & (data.matches["StartDT"] <= end_dt)
    ]
    if upcoming.empty:
        return []

    unique_dates = sorted(upcoming["Date"].unique())
    alerts: List[Dict[str, Any]] = []
    for date in unique_dates:
        forecast = weather_provider.get_forecast(
            date,
            lat=location.get("lat"),
            lon=location.get("lon"),
            city=location.get("city"),
        )
        if not forecast.is_severe:
            continue
        affected = upcoming[upcoming["Date"] == date]
        match_ids = affected["MatchID"].tolist()

        # Compute alternatives, but be careful not to put two matches in the
        # same destination slot. Track what we've handed out so each match
        # gets its own field/ref.
        alternatives = []
        already_used: list[tuple[str, str, str, str]] = []  # (date, time, field, ref)
        for _, m in affected.iterrows():
            alt = suggest_reschedule(
                data, m["MatchID"], today, _exclude_used=already_used
            )
            alternatives.append({"match_id": m["MatchID"], "alternative": alt})
            if alt:
                already_used.append(
                    (alt["date"], alt["time_start"], alt["field_id"], alt["ref_id"])
                )

        sample_pairs = ", ".join(
            f"{data.team_name(m['HomeTeamID'])} vs {data.team_name(m['AwayTeamID'])}"
            for _, m in affected.head(3).iterrows()
        )
        if len(match_ids) > 3:
            sample_pairs += f", +{len(match_ids) - 3} more"

        severity = "critical" if forecast.condition in {
            "Thunderstorm", "Tornado", "Hurricane", "Snow"
        } else "warning"

        with_alt = sum(1 for a in alternatives if a["alternative"])
        text = (
            f"Reschedule all {len(match_ids)} matches off {date} "
            f"({forecast.condition}, {forecast.precip_chance}% precip). "
            f"{with_alt}/{len(match_ids)} have an open alternative slot."
        )

        alerts.append({
            "severity": severity,
            "category": "weather",
            "title": (
                f"{forecast.icon} {forecast.condition} forecast for {date} "
                f"({forecast.precip_chance}% precip, {forecast.wind_mph}mph) "
                f"[{forecast.source}]"
            ),
            "description": (
                f"{len(match_ids)} match(es) at risk on {date}: {sample_pairs}. "
                f"{forecast.advisory}"
            ),
            "match_ids": match_ids,
            "date": date,
            "time": None,
            "suggestion": {
                "type": "weather_reschedule",
                "date": date,
                "match_alternatives": alternatives,
                "forecast": {
                    "condition": forecast.condition,
                    "temp_f": forecast.temp_f,
                    "precip_chance": forecast.precip_chance,
                    "wind_mph": forecast.wind_mph,
                    "source": forecast.source,
                    "location": forecast.location,
                },
            },
            "suggestion_text": text,
        })
    return alerts


# ------------------------------------------------------------------
# Resource availability + reschedule suggestion
# ------------------------------------------------------------------
def get_available_fields(data: LeagueData, date: str, time_start: str, ignore_match_ids=()) -> List[str]:
    booked = set(data.matches[
        (data.matches["Date"] == date) &
        (data.matches["TimeStart"] == time_start) &
        (~data.matches["MatchID"].isin(ignore_match_ids))
    ]["FieldID"].tolist())
    all_fields = sorted(data.field_names.keys())
    return [f for f in all_fields if f not in booked]


def get_available_referees(data: LeagueData, date: str, time_start: str, ignore_match_ids=()) -> List[str]:
    avail = set(data.referees[data.referees["Date"] == date]["RefID"].tolist())
    booked = set(data.matches[
        (data.matches["Date"] == date) &
        (data.matches["TimeStart"] == time_start) &
        (~data.matches["MatchID"].isin(ignore_match_ids))
    ]["RefID"].tolist())
    return sorted(avail - booked)


def suggest_reschedule(
    data: LeagueData,
    match_id: str,
    today: str,
    max_lookahead_days: int = 7,
    *,
    _exclude_used: list[tuple[str, str, str, str]] | None = None,
) -> Optional[Dict[str, Any]]:
    """Find the earliest open (date, time, field, ref) slot for a match,
    avoiding the original date. `_exclude_used` lets a caller serially
    propose alternatives for a bundle of matches without collisions."""
    used = list(_exclude_used or [])
    m = data.matches[data.matches["MatchID"] == match_id]
    if m.empty:
        return None
    m = m.iloc[0]
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    cutoff = max(today_dt, datetime.strptime(m["Date"], "%Y-%m-%d") + timedelta(days=1))
    horizon = cutoff + timedelta(days=max_lookahead_days)

    schedule = data.schedule[
        (pd.to_datetime(data.schedule["Date"]) >= cutoff) &
        (pd.to_datetime(data.schedule["Date"]) <= horizon) &
        (data.schedule["Date"] != m["Date"])
    ].sort_values(["Date", "TimeStart"])

    home_id, away_id = m["HomeTeamID"], m["AwayTeamID"]
    for _, slot in schedule.iterrows():
        date, t_start, t_end = slot["Date"], slot["TimeStart"], slot["TimeEnd"]
        # Either team already playing in this slot?
        clash = data.matches[
            (data.matches["Date"] == date) & (data.matches["TimeStart"] == t_start) &
            ((data.matches["HomeTeamID"].isin([home_id, away_id])) |
             (data.matches["AwayTeamID"].isin([home_id, away_id])))
        ]
        if not clash.empty:
            continue
        free_fields = get_available_fields(data, date, t_start, ignore_match_ids=[match_id])
        free_refs = get_available_referees(data, date, t_start, ignore_match_ids=[match_id])
        # Skip fields/refs already taken by sibling proposals in this bundle.
        used_in_slot = [(d, t, f, r) for (d, t, f, r) in used
                        if d == date and t == t_start]
        used_fields = {f for (_, _, f, _) in used_in_slot}
        used_refs = {r for (_, _, _, r) in used_in_slot}
        free_fields = [f for f in free_fields if f not in used_fields]
        free_refs = [r for r in free_refs if r not in used_refs]
        if free_fields and free_refs:
            return {
                "match_id": match_id,
                "date": date,
                "time_start": t_start,
                "time_end": t_end,
                "field_id": free_fields[0],
                "field_name": data.field_name(free_fields[0]),
                "ref_id": free_refs[0],
                "ref_name": data.referee_name(free_refs[0]),
            }
    return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _match_to_dict(data: LeagueData, m: pd.Series) -> Dict[str, Any]:
    return {
        "match_id": m["MatchID"],
        "date": m["Date"],
        "time_start": m["TimeStart"],
        "time_end": m["TimeEnd"],
        "home_team_id": m["HomeTeamID"],
        "home_team_name": data.team_name(m["HomeTeamID"]),
        "away_team_id": m["AwayTeamID"],
        "away_team_name": data.team_name(m["AwayTeamID"]),
        "field_id": m["FieldID"],
        "field_name": data.field_name(m["FieldID"]),
        "ref_id": m["RefID"],
        "ref_name": data.referee_name(m["RefID"]),
        "home_present": int(m["HomePresent"]),
        "away_present": int(m["AwayPresent"]),
    }


def _first_free_field(data: LeagueData, date: str, time_start: str,
                       exclude_field: str, ignore_match_ids) -> Optional[str]:
    free = get_available_fields(data, date, time_start, ignore_match_ids=ignore_match_ids)
    free = [f for f in free if f != exclude_field]
    return free[0] if free else None


def _first_free_slot_for_field(data: LeagueData, date: str, field_id: str,
                                exclude_match_ids) -> Optional[Dict[str, str]]:
    booked_times = set(data.matches[
        (data.matches["Date"] == date) & (data.matches["FieldID"] == field_id) &
        (~data.matches["MatchID"].isin(exclude_match_ids))
    ]["TimeStart"].tolist())
    slots = data.schedule[data.schedule["Date"] == date].sort_values("TimeStart")
    for _, s in slots.iterrows():
        if s["TimeStart"] not in booked_times:
            return {"date": date, "time_start": s["TimeStart"], "time_end": s["TimeEnd"]}
    return None


def _first_free_referee(data: LeagueData, date: str, time_start: str,
                         exclude_ref: str, ignore_match_ids) -> Optional[str]:
    free = get_available_referees(data, date, time_start, ignore_match_ids=ignore_match_ids)
    free = [r for r in free if r != exclude_ref]
    return free[0] if free else None


# ------------------------------------------------------------------
# Tool registry — used by the LLM-driven agent
# ------------------------------------------------------------------
def build_tool_registry(
    data: LeagueData,
    *,
    weather_provider: WeatherProvider | None = None,
    location: Dict[str, Any] | None = None,
):
    """Returns {tool_name: callable} that the LLM can invoke.

    Weather provider and location can be updated at runtime by mutating the
    closure-captured `_runtime` dict — see `LeagueManagerAgent.set_location`.
    """
    _runtime: Dict[str, Any] = {
        "weather": weather_provider or MockWeatherProvider(),
        "location": location or {"lat": None, "lon": None, "city": "Demo City"},
    }

    def _check_weather(today, days=14):
        return check_weather_alerts(data, today, _runtime["weather"], _runtime["location"], days)

    registry = {
        "get_upcoming_matches": lambda today, days=14: get_upcoming_matches(data, today, days),
        "get_match_details": lambda match_id: get_match_details(data, match_id),
        "get_team_roster": lambda team_id: get_team_roster(data, team_id),
        "get_team_matches": lambda team_id, today=None, limit=5: get_team_matches(data, team_id, today, limit),
        "get_standings": lambda today, top=12: get_standings(data, today, top),
        "get_standby_players": lambda team_id: get_standby_players(data, team_id),
        "check_field_conflicts": lambda today: check_field_conflicts(data, today),
        "check_referee_conflicts": lambda today: check_referee_conflicts(data, today),
        "check_roster_shortage": lambda today: check_roster_shortage(data, today),
        "check_rematch_violations": lambda today: check_rematch_violations(data, today),
        "check_weekly_game_limit": lambda today: check_weekly_game_limit(data, today),
        "check_weather_alerts": _check_weather,
        "get_available_fields": lambda date, time_start: get_available_fields(data, date, time_start),
        "get_available_referees": lambda date, time_start: get_available_referees(data, date, time_start),
        "suggest_reschedule": lambda match_id, today: suggest_reschedule(data, match_id, today),
    }
    # Expose the runtime dict so the agent can swap provider / location later.
    registry["_runtime"] = _runtime  # type: ignore[assignment]
    return registry


TOOL_DESCRIPTIONS = [
    {"name": "get_upcoming_matches",
     "args": {"today": "YYYY-MM-DD", "days": "int (default 14)"},
     "doc": "List ALL matches scheduled within `days` from today."},
    {"name": "get_match_details",
     "args": {"match_id": "string e.g. 'M1377'"},
     "doc": "Full record for a single match by ID."},
    {"name": "get_team_roster",
     "args": {"team_id": "string e.g. 'T07' or team name e.g. 'Dragons'"},
     "doc": "Starters, standby list, and total players for a team."},
    {"name": "get_team_matches",
     "args": {"team_id": "string", "today": "YYYY-MM-DD (optional)", "limit": "int (default 5)"},
     "doc": "Matches involving a specific team. With `today` set, returns ONLY upcoming matches. Use this when the user asks 'when does team X play next?'."},
    {"name": "get_standings",
     "args": {"today": "YYYY-MM-DD"},
     "doc": "League standings: W/L/T/PTS for every team, ranked. Use this for 'who is leading?' / 'show standings'."},
    {"name": "get_standby_players",
     "args": {"team_id": "string"},
     "doc": "Standby/substitute players (the bench) for a team."},
    {"name": "check_field_conflicts", "args": {"today": "YYYY-MM-DD"},
     "doc": "Find double-booked fields with reassignment suggestions."},
    {"name": "check_referee_conflicts", "args": {"today": "YYYY-MM-DD"},
     "doc": "Find double-booked referees with reassignment suggestions."},
    {"name": "check_roster_shortage", "args": {"today": "YYYY-MM-DD"},
     "doc": "Find teams below the 9-player minimum with call-up suggestions."},
    {"name": "check_rematch_violations", "args": {"today": "YYYY-MM-DD"},
     "doc": "Pairs scheduled to play each other too often (only when rule is enabled)."},
    {"name": "check_weekly_game_limit", "args": {"today": "YYYY-MM-DD"},
     "doc": "Teams with too many games in a week (only when rule is enabled)."},
    {"name": "check_weather_alerts", "args": {"today": "YYYY-MM-DD", "days": "int (default 14)"},
     "doc": "Weather-driven reschedule alerts, one per severe day."},
    {"name": "get_available_fields", "args": {"date": "YYYY-MM-DD", "time_start": "e.g. '05:00 pm'"},
     "doc": "Free fields in a given slot."},
    {"name": "get_available_referees", "args": {"date": "YYYY-MM-DD", "time_start": "e.g. '05:00 pm'"},
     "doc": "Free referees in a given slot."},
    {"name": "suggest_reschedule", "args": {"match_id": "string", "today": "YYYY-MM-DD"},
     "doc": "Concrete reschedule proposal: date, time, field, referee."},
]
