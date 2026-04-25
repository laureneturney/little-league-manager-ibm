"""Computes win/loss/tie standings from completed matches.

Rules:
  * A match is "completed" when its date is strictly before `today`.
  * If both teams have < 9 players present  -> double forfeit (loss for both).
  * If one team has < 9 players present     -> the short team forfeits.
  * Otherwise the team with more attendees wins; equal attendance with both
    >= 9 is treated as a tie. This makes attendance meaningful while
    keeping outcomes deterministic from the dataset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .data_loader import LeagueData

MIN_PLAYERS = 9


@dataclass
class TeamRecord:
    team_id: str
    team_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    forfeits_for: int = 0   # times this team forfeited
    forfeits_against: int = 0  # wins by forfeit
    games_played: int = 0
    attendance_avg: float = 0.0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.ties

    @property
    def win_pct(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0


@dataclass
class MatchOutcome:
    match_id: str
    date: str
    home_id: str
    away_id: str
    home_present: int
    away_present: int
    winner_id: str | None  # None => tie
    forfeit: bool
    note: str


def compute_standings(data: LeagueData, today: str) -> Dict[str, TeamRecord]:
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    records: Dict[str, TeamRecord] = {
        tid: TeamRecord(team_id=tid, team_name=name)
        for tid, name in data.teams.items()
    }
    attendance_totals: Dict[str, list[int]] = {tid: [] for tid in records}

    for _, m in data.matches.iterrows():
        match_date = datetime.strptime(m["Date"], "%Y-%m-%d")
        if match_date >= today_dt:
            continue  # not played yet

        outcome = _resolve(m)
        h, a = m["HomeTeamID"], m["AwayTeamID"]

        records[h].games_played += 1
        records[a].games_played += 1
        attendance_totals[h].append(int(m["HomePresent"]))
        attendance_totals[a].append(int(m["AwayPresent"]))

        if outcome.forfeit:
            if outcome.winner_id is None:
                # double forfeit
                records[h].losses += 1
                records[a].losses += 1
                records[h].forfeits_for += 1
                records[a].forfeits_for += 1
            else:
                loser = a if outcome.winner_id == h else h
                records[outcome.winner_id].wins += 1
                records[outcome.winner_id].forfeits_against += 1
                records[loser].losses += 1
                records[loser].forfeits_for += 1
        else:
            if outcome.winner_id is None:
                records[h].ties += 1
                records[a].ties += 1
            else:
                loser = a if outcome.winner_id == h else h
                records[outcome.winner_id].wins += 1
                records[loser].losses += 1

    for tid, atts in attendance_totals.items():
        if atts:
            records[tid].attendance_avg = round(sum(atts) / len(atts), 1)

    return records


def _resolve(m: pd.Series) -> MatchOutcome:
    h, a = m["HomeTeamID"], m["AwayTeamID"]
    hp, ap = int(m["HomePresent"]), int(m["AwayPresent"])
    if hp < MIN_PLAYERS and ap < MIN_PLAYERS:
        return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap,
                            winner_id=None, forfeit=True,
                            note="Double forfeit — both rosters short.")
    if hp < MIN_PLAYERS:
        return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap,
                            winner_id=a, forfeit=True,
                            note=f"{h} forfeits ({hp} players).")
    if ap < MIN_PLAYERS:
        return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap,
                            winner_id=h, forfeit=True,
                            note=f"{a} forfeits ({ap} players).")
    if hp > ap:
        return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap, h, False,
                            f"{h} wins on attendance ({hp}-{ap}).")
    if ap > hp:
        return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap, a, False,
                            f"{a} wins on attendance ({ap}-{hp}).")
    return MatchOutcome(m["MatchID"], m["Date"], h, a, hp, ap, None, False,
                        f"Tie ({hp}-{ap}).")


def standings_table(data: LeagueData, today: str) -> List[dict]:
    records = compute_standings(data, today)
    rows = []
    for r in records.values():
        rows.append({
            "Team": f"{r.team_name} ({r.team_id})",
            "GP": r.games_played,
            "W": r.wins,
            "L": r.losses,
            "T": r.ties,
            "PTS": r.points,
            "Win %": round(r.win_pct, 3),
            "Forfeits": r.forfeits_for,
            "Avg Attendance": r.attendance_avg,
        })
    rows.sort(key=lambda x: (-x["PTS"], -x["W"], -x["Win %"], x["Team"]))
    return rows
