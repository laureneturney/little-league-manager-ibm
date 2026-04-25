"""The Little League Manager Agent.

Two surfaces:
  1. `run_diagnostics(today)` — autonomous monitoring pass. Pure rules, no LLM
     required. Always available, even with `LLM_PROVIDER=mock`. This is what
     drives the dashboard, weather, conflicts and roster tabs.

  2. `chat(user_message, today)` — free-form Q&A using a ReAct-style tool loop.
     With `watsonx` or `custom` providers the LLM picks tools and synthesizes
     the answer. With `mock` we fall back to a deterministic keyword-based
     responder so the demo works end-to-end without any API key.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .data_loader import LeagueData
from .llm_provider import LLMProvider
from .tools import (
    TOOL_DESCRIPTIONS,
    build_tool_registry,
    check_field_conflicts,
    check_referee_conflicts,
    check_roster_shortage,
    check_rematch_violations,
    check_weekly_game_limit,
    check_weather_alerts,
    get_upcoming_matches,
    suggest_reschedule,
)
from .weather import MockWeatherProvider, WeatherProvider


@dataclass
class DiagnosticsReport:
    today: str
    upcoming_matches: List[Dict[str, Any]]
    field_conflicts: List[Dict[str, Any]]
    referee_conflicts: List[Dict[str, Any]]
    roster_shortages: List[Dict[str, Any]]
    rematch_violations: List[Dict[str, Any]]
    weekly_limit_violations: List[Dict[str, Any]]
    weather_alerts: List[Dict[str, Any]]

    @property
    def all_alerts(self) -> List[Dict[str, Any]]:
        return (
            self.field_conflicts
            + self.referee_conflicts
            + self.roster_shortages
            + self.weather_alerts
            + self.rematch_violations
            + self.weekly_limit_violations
        )

    def summary_counts(self) -> Dict[str, int]:
        return {
            "critical": sum(1 for a in self.all_alerts if a["severity"] == "critical"),
            "warning": sum(1 for a in self.all_alerts if a["severity"] == "warning"),
            "info": sum(1 for a in self.all_alerts if a["severity"] == "info"),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You are the Little League Manager — an autonomous scheduling \
agent for a youth sports league. You monitor field availability, referee \
assignments, rosters (9-player minimum), and weather. When you find a problem, \
you propose a SPECIFIC, actionable resolution: a Field number, a time slot, or \
a Referee name. Your suggestions will be ingested by a UI, so always be concrete.

You may call tools to query the league database. Respond with ONE JSON object \
on each turn, in one of these two shapes:

  {"tool": "<tool_name>", "args": { ... }}             # call a tool
  {"final_answer": "<plain-text answer for the user>"}  # finish

Available tools:
%s

Rules:
- Always cite specific Match IDs, Field IDs (F01..F03), Referee IDs (R01..R08), \
and exact times.
- Stop and emit `final_answer` as soon as you have enough information.
- Never invent data. If a tool returns nothing, say so.
"""


class LeagueManagerAgent:
    def __init__(
        self,
        data: LeagueData,
        llm: LLMProvider,
        weather_provider: WeatherProvider | None = None,
        location: Dict[str, Any] | None = None,
        *,
        max_steps: int = 6,
    ):
        self.data = data
        self.llm = llm
        self.weather = weather_provider or MockWeatherProvider()
        self._location = location or {"lat": None, "lon": None, "city": "Demo City"}
        self.rule_limits: Dict[str, int | None] = {
            "max_rematches": None,         # None => rule disabled
            "max_games_per_week": None,    # None => rule disabled
        }
        self.tools = build_tool_registry(data, weather_provider=self.weather, location=self._location)
        self.max_steps = max_steps
        self.system_prompt = SYSTEM_PROMPT % json.dumps(TOOL_DESCRIPTIONS, indent=2)

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------
    @property
    def location(self) -> Dict[str, Any]:
        return dict(self._location)

    def set_location(self, *, lat: float | None, lon: float | None, city: str) -> None:
        self._location.update({"lat": lat, "lon": lon, "city": city})
        # tool registry holds a reference to the same dict via `_runtime`
        self.tools["_runtime"]["location"] = self._location  # type: ignore[index]

    def set_weather_provider(self, provider: WeatherProvider) -> None:
        self.weather = provider
        self.tools["_runtime"]["weather"] = provider  # type: ignore[index]

    def set_rule_limits(
        self, *, max_rematches: int | None = None, max_games_per_week: int | None = None,
    ) -> None:
        """Set caps for the rematch and weekly-game-limit rules. `None` (or 0)
        disables the rule, in which case it produces no alerts."""
        self.rule_limits["max_rematches"] = max_rematches if (max_rematches or 0) > 0 else None
        self.rule_limits["max_games_per_week"] = (
            max_games_per_week if (max_games_per_week or 0) > 0 else None
        )

    # ------------------------------------------------------------------
    # Autonomous monitoring (no LLM required)
    # ------------------------------------------------------------------
    def run_diagnostics(self, today: str) -> DiagnosticsReport:
        return DiagnosticsReport(
            today=today,
            upcoming_matches=get_upcoming_matches(self.data, today, days=7),
            field_conflicts=check_field_conflicts(self.data, today),
            referee_conflicts=check_referee_conflicts(self.data, today),
            roster_shortages=check_roster_shortage(self.data, today),
            rematch_violations=check_rematch_violations(
                self.data, today, self.rule_limits["max_rematches"]
            ),
            weekly_limit_violations=check_weekly_game_limit(
                self.data, today, self.rule_limits["max_games_per_week"]
            ),
            weather_alerts=check_weather_alerts(self.data, today, self.weather, self._location, days=14),
        )

    # ------------------------------------------------------------------
    # Free-form chat (ReAct tool loop, falls back to mock responder)
    # ------------------------------------------------------------------
    def chat(self, user_message: str, today: str) -> Dict[str, Any]:
        if self.llm.name == "mock":
            return self._mock_chat(user_message, today)
        return self._llm_chat(user_message, today)

    def _llm_chat(self, user_message: str, today: str) -> Dict[str, Any]:
        scratchpad = [f"Today is {today}.", f"User question: {user_message}", ""]
        trace: List[Dict[str, Any]] = []
        for step in range(self.max_steps):
            response = self.llm.complete(self.system_prompt, "\n".join(scratchpad))
            parsed = self._parse_json(response)
            if not parsed:
                return {
                    "answer": (
                        "The agent's response could not be parsed. Raw output:\n\n"
                        + response
                    ),
                    "trace": trace,
                    "provider": self.llm.name,
                }
            if "final_answer" in parsed:
                return {"answer": parsed["final_answer"], "trace": trace, "provider": self.llm.name}
            tool = parsed.get("tool")
            args = parsed.get("args", {}) or {}
            if tool not in self.tools:
                scratchpad.append(f"Tool `{tool}` not found. Available: {list(self.tools)}")
                continue
            try:
                result = self.tools[tool](**args)
            except TypeError as e:
                scratchpad.append(f"Tool `{tool}` argument error: {e}")
                continue
            trace.append({"step": step + 1, "tool": tool, "args": args, "result_preview": _preview(result)})
            scratchpad.append(f"Tool `{tool}` returned: {json.dumps(result, default=str)[:1500]}")
        return {
            "answer": "Reached step limit before producing a final answer. Tool trace below.",
            "trace": trace,
            "provider": self.llm.name,
        }

    def _mock_chat(self, user_message: str, today: str) -> Dict[str, Any]:
        """Deterministic responder used when no LLM is configured."""
        msg = user_message.lower()
        report = self.run_diagnostics(today)
        chunks: List[str] = []

        if any(k in msg for k in ("conflict", "double", "overbook")):
            chunks.append(_render_alerts(
                "Field conflicts", report.field_conflicts) or
                "No field conflicts detected.")
            chunks.append(_render_alerts(
                "Referee conflicts", report.referee_conflicts) or
                "No referee conflicts detected.")
        if any(k in msg for k in ("weather", "rain", "storm")):
            chunks.append(_render_alerts(
                "Weather alerts", report.weather_alerts) or
                "No severe weather in the next 14 days.")
        if any(k in msg for k in ("roster", "player", "shortage", "forfeit", "substitute", "standby")):
            chunks.append(_render_alerts(
                "Roster shortages", report.roster_shortages) or
                "All teams meet the 9-player minimum.")
        if any(k in msg for k in ("rematch", "limit", "twice", "weekly")):
            chunks.append(_render_alerts("Schedule violations",
                report.rematch_violations + report.weekly_limit_violations))
        if any(k in msg for k in ("schedule", "upcoming", "today", "tomorrow", "next")):
            chunks.append(_render_upcoming(report.upcoming_matches))

        # Specific match query: "M1377", "match m1377", etc.
        m = re.search(r"M?\s*(\d{4})", user_message.upper())
        if m:
            mid = "M" + m.group(1)
            details = self.tools["get_match_details"](match_id=mid)
            if details:
                alt = suggest_reschedule(self.data, mid, today)
                chunks.append(_render_match(details, alt))

        if not chunks:
            counts = report.summary_counts()
            top = report.all_alerts[:5]
            chunks.append(
                f"I checked the league for you. Today is {today}.\n\n"
                f"- {counts['critical']} critical alert(s)\n"
                f"- {counts['warning']} warning(s)\n"
                f"- {counts['info']} info item(s)\n"
                f"- {len(report.upcoming_matches)} match(es) in the next 7 days"
            )
            if top:
                chunks.append(_render_alerts("Top alerts", top))

        return {
            "answer": "\n\n".join(c for c in chunks if c),
            "trace": [],
            "provider": "mock",
        }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any] | None:
        """Tolerant JSON extraction — LLMs love to wrap JSON in prose / fences."""
        text = text.strip()
        # Strip ```json ... ``` fences if present
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        # Otherwise grab first { ... } block
        else:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                text = brace.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


# ----------------------------------------------------------------------
# Render helpers (used by mock chat)
# ----------------------------------------------------------------------
def _render_alerts(heading: str, alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return ""
    lines = [f"### {heading} ({len(alerts)})"]
    for a in alerts[:6]:
        sev = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(a["severity"], "⚪")
        lines.append(f"- {sev} **{a['title']}**\n  {a['description']}\n  → _{a['suggestion_text']}_")
    if len(alerts) > 6:
        lines.append(f"_…and {len(alerts) - 6} more._")
    return "\n".join(lines)


def _render_upcoming(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No matches scheduled in the next 7 days."
    lines = [f"### Upcoming matches ({len(matches)})"]
    for m in matches[:8]:
        lines.append(
            f"- **{m['date']} {m['time_start']}** — "
            f"{m['home_team_name']} vs {m['away_team_name']} • "
            f"{m['field_name']} • Ref: {m['ref_name']} "
            f"[{m['home_present']}/{m['away_present']} attending]"
        )
    if len(matches) > 8:
        lines.append(f"_…and {len(matches) - 8} more._")
    return "\n".join(lines)


def _render_match(m: Dict[str, Any], alt: Dict[str, Any] | None) -> str:
    base = (
        f"### Match {m['match_id']}\n"
        f"- {m['date']} {m['time_start']}–{m['time_end']}\n"
        f"- {m['home_team_name']} (Home) vs {m['away_team_name']} (Away)\n"
        f"- Field: {m['field_name']} ({m['field_id']})\n"
        f"- Referee: {m['ref_name']} ({m['ref_id']})\n"
        f"- Attendance: {m['home_present']} / {m['away_present']}"
    )
    if alt:
        base += (
            f"\n\n**If reschedule needed:** "
            f"{alt['date']} {alt['time_start']} on {alt['field_name']} "
            f"with {alt['ref_name']}."
        )
    return base


def _preview(result: Any, limit: int = 240) -> str:
    s = json.dumps(result, default=str)
    return s[:limit] + ("…" if len(s) > limit else "")
