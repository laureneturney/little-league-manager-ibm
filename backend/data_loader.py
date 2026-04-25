"""Loads the five league CSVs into pandas DataFrames + helpful indexes."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def _parse_time(t: str) -> datetime:
    """Parses '05:00 pm' / '11:00 am' regardless of case."""
    return datetime.strptime(t.strip().upper(), "%I:%M %p")


def _to_24h(t: str) -> str:
    """'05:00 pm' -> '17:00'. Used for sorting and display."""
    return _parse_time(t).strftime("%H:%M")


def _datetime_at(date_str: str, time_str: str) -> datetime:
    return datetime.combine(
        datetime.strptime(date_str, "%Y-%m-%d").date(),
        _parse_time(time_str).time(),
    )


@dataclass
class LeagueData:
    matches: pd.DataFrame
    fields: pd.DataFrame
    referees: pd.DataFrame
    schedule: pd.DataFrame
    roster: pd.DataFrame
    teams: Dict[str, str] = field(default_factory=dict)        # T01 -> "Strikers"
    field_names: Dict[str, str] = field(default_factory=dict)  # F01 -> "Field 1"
    referee_names: Dict[str, str] = field(default_factory=dict)  # R01 -> "John Smith"

    def team_name(self, team_id: str) -> str:
        return self.teams.get(team_id, team_id)

    def field_name(self, field_id: str) -> str:
        return self.field_names.get(field_id, field_id)

    def referee_name(self, ref_id: str) -> str:
        return self.referee_names.get(ref_id, ref_id)


def load_all(data_dir: Path) -> LeagueData:
    matches = pd.read_csv(data_dir / "matches_final.csv", dtype=str)
    fields = pd.read_csv(data_dir / "fields_final.csv", dtype=str)
    referees = pd.read_csv(data_dir / "referees_final.csv", dtype=str)
    schedule = pd.read_csv(data_dir / "league_schedule_final.csv", dtype=str)
    roster = pd.read_csv(data_dir / "roster_final.csv", dtype=str)

    matches["HomePresent"] = matches["HomePresent"].astype(int)
    matches["AwayPresent"] = matches["AwayPresent"].astype(int)
    matches["StartDT"] = matches.apply(lambda r: _datetime_at(r["Date"], r["TimeStart"]), axis=1)
    matches["EndDT"] = matches.apply(lambda r: _datetime_at(r["Date"], r["TimeEnd"]), axis=1)
    matches["Time24"] = matches["TimeStart"].map(_to_24h)
    matches["TimeSlot"] = matches["TimeStart"] + " - " + matches["TimeEnd"]

    fields["StartDT"] = fields.apply(lambda r: _datetime_at(r["Date"], r["BookedStart"]), axis=1)
    fields["EndDT"] = fields.apply(lambda r: _datetime_at(r["Date"], r["BookedEnd"]), axis=1)

    teams = dict(zip(roster["TeamID"], roster["TeamName"]))
    field_names = dict(zip(fields["FieldID"], fields["FieldName"]))
    referee_names = dict(zip(referees["RefID"], referees["RefereeName"]))

    return LeagueData(
        matches=matches.reset_index(drop=True),
        fields=fields.reset_index(drop=True),
        referees=referees.reset_index(drop=True),
        schedule=schedule.reset_index(drop=True),
        roster=roster.reset_index(drop=True),
        teams=teams,
        field_names=field_names,
        referee_names=referee_names,
    )


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")
