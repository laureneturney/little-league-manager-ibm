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


SYSTEM_PROMPT_TEMPLATE = """You are the Little League Manager — an autonomous \
scheduling agent for a youth sports league. Answer the league coordinator's \
questions using ONLY the GROUND-TRUTH FACTS and the TOOLS below. Never invent \
team names, players, fields, or referees.

GROUND-TRUTH FACTS (always accurate, do not contradict):
- Today's date: {today}
- League location: {location}
- Teams (12 total): {teams}
- Fields (3 total): {fields}
- Referees (8 total): {referees}
- Roster minimum is 9 players; below that the team forfeits.

TOOLS available to you:
{tools}

INTERACTION FORMAT — STRICT:
On every turn, output EXACTLY ONE JSON object. Nothing else. No prose, no \
markdown, no code fences. The two valid shapes are:

  {{"tool": "<tool_name>", "args": {{ "<arg>": <value>, ... }}}}
  {{"final_answer": "<plain text for the user>"}}

NEVER emit two JSON objects in one turn. NEVER call a tool and provide a \
final_answer in the same response.

Workflow:
1. If the user's question is fully answered by the GROUND-TRUTH FACTS \
(e.g., "what teams are in the league?"), reply with a final_answer immediately.
2. Otherwise call ONE tool, observe its result on the next turn, then either \
call another tool or emit a final_answer.
3. Use real values from the ground-truth list — e.g., the Dragons are T07, \
not "All-Stars" or "Cardinals". Field IDs are F01..F03 only. Referee IDs are \
R01..R08 only.
4. Cite specific Match IDs, dates, and times when answering scheduling questions.

EXAMPLES:

User: What teams are in the league?
You: {{"final_answer": "The 12 teams are Strikers (T01), Titans (T02), Eagles (T03), Thunder (T04), United (T05), Wolves (T06), Dragons (T07), Lions (T08), Hawks (T09), Blazers (T10), Cobras (T11), and Storm (T12)."}}

User: When do the Dragons play next?
You: {{"tool": "get_team_matches", "args": {{"team_id": "T07", "today": "{today}", "limit": 1}}}}
[tool result observed]
You: {{"final_answer": "The Dragons (T07) play next on 2026-04-25 at 1:00pm vs the Hawks (T09) on Field 3, refereed by Karen Davis."}}

User: Who is leading the league?
You: {{"tool": "get_standings", "args": {{"today": "{today}"}}}}
[tool result observed]
You: {{"final_answer": "The Dragons (T07) are leading with 35-23-6 (111 points), followed by the Cobras (T11) at 31-23-10."}}
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
        system_prompt = self._build_system_prompt(today)
        scratchpad = [f"User question: {user_message}", ""]
        trace: List[Dict[str, Any]] = []
        consecutive_parse_failures = 0

        for step in range(self.max_steps):
            response = self.llm.complete(system_prompt, "\n".join(scratchpad))
            parsed = self._parse_first_json(response)

            if not parsed:
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= 2:
                    # Bail out and use the deterministic responder.
                    fallback = self._mock_chat(user_message, today)
                    fallback["trace"] = trace
                    fallback["provider"] = self.llm.name + " (fallback to rules)"
                    return fallback
                scratchpad.append(
                    "Your previous output could not be parsed as JSON. Output "
                    "EXACTLY ONE JSON object — no prose, no markdown, no fences."
                )
                continue
            consecutive_parse_failures = 0

            if "final_answer" in parsed and "tool" not in parsed:
                return {"answer": parsed["final_answer"], "trace": trace, "provider": self.llm.name}

            tool = parsed.get("tool")
            args = parsed.get("args", {}) or {}
            if tool not in self.tools:
                scratchpad.append(
                    f"Tool `{tool}` does not exist. Choose from: {sorted(self.tools)}"
                )
                continue

            # Accept both list (positional) and dict (keyword) args.
            try:
                if isinstance(args, list):
                    result = self.tools[tool](*args)
                elif isinstance(args, dict):
                    result = self.tools[tool](**args)
                else:
                    result = self.tools[tool]()
            except TypeError as e:
                scratchpad.append(f"Tool `{tool}` argument error: {e}. Re-issue the call with correct args.")
                continue
            except Exception as e:  # noqa: BLE001
                scratchpad.append(f"Tool `{tool}` failed: {e}. Try a different approach.")
                continue

            trace.append({
                "step": step + 1, "tool": tool, "args": args,
                "result_preview": _preview(result),
            })
            scratchpad.append(
                f"Result of `{tool}`: {json.dumps(result, default=str)[:1500]}\n\n"
                f"Now produce ONE JSON object: either another tool call OR a final_answer "
                f"that answers the user's question using this result."
            )

        # Step limit hit — fall back to rules so the user sees a useful answer.
        fallback = self._mock_chat(user_message, today)
        fallback["trace"] = trace
        fallback["provider"] = self.llm.name + " (fallback to rules)"
        return fallback

    def _build_system_prompt(self, today: str) -> str:
        teams = ", ".join(f"{tid} {name}" for tid, name in sorted(self.data.teams.items()))
        fields = ", ".join(f"{fid} ({name})" for fid, name in sorted(self.data.field_names.items()))
        referees = ", ".join(f"{rid} ({name})" for rid, name in sorted(self.data.referee_names.items()))
        location = self._location.get("city") or "(not set)"
        tools_doc = json.dumps(TOOL_DESCRIPTIONS, indent=2)
        return SYSTEM_PROMPT_TEMPLATE.format(
            today=today,
            location=location,
            teams=teams,
            fields=fields,
            referees=referees,
            tools=tools_doc,
        )

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
    def _parse_first_json(text: str) -> Dict[str, Any] | None:
        """Return the first balanced + parseable {...} object found in `text`.

        Robust to multiple objects (`{...} {...}`), code fences, prose around
        the object, and braces inside string values. Quote/escape-aware brace
        counting — a previous regex-based version would greedily span across
        multiple objects and fail to parse.
        """
        if not text:
            return None
        # Drop leading whitespace and obvious code fences
        stripped = text.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
        if fence:
            stripped = fence.group(1)

        depth = 0
        start: int | None = None
        in_string = False
        escape = False
        for i, ch in enumerate(stripped):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = stripped[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None  # try the next object
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
