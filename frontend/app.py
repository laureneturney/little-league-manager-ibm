"""Little League Manager — Streamlit frontend.

Layout matches the spec: a Dashboard landing page with a 2-column card grid
(Upcoming Schedule / Conflict Alerts / AI Suggestions / Recent Notifications /
Weather Forecast), plus six lightweight detail pages reachable from the cards
or the sidebar nav.

Run:  streamlit run frontend/app.py
"""
from __future__ import annotations
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import actions, cancellations as cancel, mutations  # noqa: E402
from backend.agent import LeagueManagerAgent  # noqa: E402
from backend.config import Config  # noqa: E402
from backend.data_loader import load_all  # noqa: E402
from backend.llm_provider import get_provider as get_llm_provider  # noqa: E402
from backend.standings import standings_table  # noqa: E402
from backend.weather import (  # noqa: E402
    FORCED_BAD_WEATHER,
    OpenMeteoProvider,
    OpenWeatherProvider,
    get_provider as get_weather_provider,
)


# ----------------------------------------------------------------------
# Setup + constants
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="League Management Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

LLM_DISPLAY_NAME = "IBM watsonx"

PAGES: dict[str, str] = {
    "dashboard": "🏠 Dashboard",
    "schedule": "📅 Schedule",
    "alerts": "⚠️ Alerts",
    "suggestions": "✨ AI Suggestions",
    "notifications": "📬 Notifications",
    "weather": "🌤️ Weather",
    "chat": "💬 Agent Chat",
}

SEVERITY_BADGE = {
    "critical": ("🔴", "#9b1c1c"),
    "warning":  ("🟡", "#92400e"),
    "info":     ("🔵", "#1e40af"),
}


# ----------------------------------------------------------------------
# Boot
# ----------------------------------------------------------------------
@st.cache_resource
def boot():
    cfg = Config.from_env()
    data = load_all(cfg.data_dir)
    # Replay any persisted mutations so the live DataFrame reflects prior
    # Confirm clicks across Streamlit restarts.
    mutations.replay(data, cfg.data_dir)
    return cfg, data


@st.cache_resource
def get_agent():
    cfg, data = boot()
    try:
        llm = get_llm_provider(cfg)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(
            f"Failed to init configured LLM provider: {e}\nFalling back to mock."
        )
        cfg.llm_provider = "mock"
        llm = get_llm_provider(cfg)
    weather = get_weather_provider(cfg)
    if cfg.weather_provider == "openweather" and not isinstance(weather, OpenWeatherProvider):
        st.sidebar.warning(
            "WEATHER_PROVIDER=openweather but OPENWEATHER_API_KEY is empty. Using mock."
        )
    agent = LeagueManagerAgent(
        data=data,
        llm=llm,
        weather_provider=weather,
        location={"lat": cfg.league_lat, "lon": cfg.league_lon, "city": cfg.league_city},
    )
    return agent, cfg, data


cfg_initial, data = boot()
agent, cfg, _ = get_agent()

if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "loc" not in st.session_state:
    st.session_state.loc = {
        "city": cfg.league_city,
        "lat": cfg.league_lat,
        "lon": cfg.league_lon,
    }
if "diagnostics_run_at" not in st.session_state:
    st.session_state.diagnostics_run_at = datetime.now()

agent.set_location(**st.session_state.loc)


def goto(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def fmt_date(s: str) -> str:
    """2026-04-25 → 'Sat, Apr 25'."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%a, %b %d")
    except ValueError:
        return s


def time_ago(dt: datetime) -> str:
    diff = (datetime.now() - dt).total_seconds()
    if diff < 60:
        return "just now"
    if diff < 3600:
        m = int(diff / 60)
        return f"{m} min{'s' if m != 1 else ''} ago"
    h = int(diff / 3600)
    return f"{h} hr{'s' if h != 1 else ''} ago"


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("⚾ Little League Manager")
st.sidebar.caption("Autonomous scheduling agent")

st.sidebar.markdown("### Navigation")
for key, label in PAGES.items():
    btn_type = "primary" if st.session_state.page == key else "secondary"
    if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
        st.session_state.page = key
        st.rerun()

st.sidebar.markdown("---")
default_today = cfg_initial.demo_today or date.today().isoformat()
today_input = st.sidebar.date_input(
    "Today (for monitoring)",
    value=pd.to_datetime(default_today).date(),
    help="Pin a date for reproducible demos. Dataset spans 2026-01-15 — 2026-05-05.",
)
today = today_input.isoformat()

st.sidebar.subheader("📍 League location")
with st.sidebar.form("loc_form", clear_on_submit=False):
    city_input = st.text_input("City", value=st.session_state.loc["city"])
    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Lat", value=float(st.session_state.loc["lat"]),
                                format="%.4f", step=0.01)
    lon_input = c2.number_input("Lon", value=float(st.session_state.loc["lon"]),
                                format="%.4f", step=0.01)
    submitted = st.form_submit_button("Apply", use_container_width=True)
    geocode_clicked = st.form_submit_button(
        "🔍 Geocode city → lat/lon", use_container_width=True
    )

if geocode_clicked:
    if isinstance(agent.weather, (OpenWeatherProvider, OpenMeteoProvider)):
        result = agent.weather.geocode(city_input.strip())
        if result:
            st.session_state.loc = {"city": result[2], "lat": result[0], "lon": result[1]}
            st.rerun()
        else:
            st.sidebar.error("Could not resolve that city.")
    else:
        st.sidebar.warning("Geocoding requires a real weather provider.")

if submitted:
    st.session_state.loc = {
        "city": city_input.strip() or "Unknown",
        "lat": float(lat_input),
        "lon": float(lon_input),
    }
    st.rerun()

st.sidebar.caption(
    f"📌 **{st.session_state.loc['city']}** "
    f"({st.session_state.loc['lat']:.3f}, {st.session_state.loc['lon']:.3f})"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Rule limits")
st.sidebar.caption("0 = rule disabled (no alerts).")
max_rematches = st.sidebar.number_input(
    "Rematches max (same pair)", min_value=0, max_value=20,
    value=int(st.session_state.get("max_rematches", 0)), step=1,
    help="Alert when any team pair is scheduled to play each other this many times or more.",
)
max_weekly = st.sidebar.number_input(
    "Weekly games max (per team)", min_value=0, max_value=14,
    value=int(st.session_state.get("max_weekly", 0)), step=1,
    help="Alert when a team has this many or more games in a single Mon–Sun week.",
)
st.session_state.max_rematches = int(max_rematches)
st.session_state.max_weekly = int(max_weekly)
agent.set_rule_limits(
    max_rematches=int(max_rematches),
    max_games_per_week=int(max_weekly),
)

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Reset all schedule mutations", use_container_width=True,
                     help="Clears mutations.json and reloads the original CSV data on next render."):
    mutations.clear_all(cfg.data_dir)
    st.cache_resource.clear()
    st.rerun()

st.sidebar.success(f"LLM: **{LLM_DISPLAY_NAME}** · Weather: **{agent.weather.name}**")


# ----------------------------------------------------------------------
# Run diagnostics for this render
# ----------------------------------------------------------------------
report = agent.run_diagnostics(today)
st.session_state.diagnostics_run_at = datetime.now()
cancelled_ids = cancel.cancelled_match_ids(cfg.data_dir)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def card_header(title: str, link_label: str, link_target: str, container) -> None:
    cols = container.columns([4, 1])
    cols[0].markdown(f"### {title}")
    if cols[1].button(link_label, key=f"link_{link_target}_{title}",
                      use_container_width=True):
        goto(link_target)


def render_alert_pretty(alert: dict) -> None:
    icon, fg = SEVERITY_BADGE.get(alert["severity"], ("⚪", "#333"))
    bg = {"critical": "#FFE5E5", "warning": "#FFF7DA", "info": "#E0F0FF"}.get(
        alert["severity"], "#eee"
    )
    st.markdown(
        f"""
        <div style="background:{bg};color:{fg};padding:14px 16px;
                    border-radius:10px;margin-bottom:8px;">
          <div style="font-weight:700;font-size:1.02em">{icon} {alert['title']}</div>
          <div style="margin-top:6px">{alert['description']}</div>
          <div style="margin-top:10px;padding:8px 10px;background:rgba(0,0,0,0.06);border-radius:6px">
            <b>Suggested resolution:</b> {alert['suggestion_text']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confirm_suggestion(alert: dict) -> bool:
    """Apply a suggestion to the live schedule and log the action.

    Returns True if at least one operation was applied. The caller should
    `st.rerun()` so all tabs reflect the mutation.
    """
    record = mutations.apply_suggestion(data, cfg.data_dir, alert)
    if not record["ops"]:
        st.toast("Nothing to apply for this suggestion.", icon="⚠️")
        return False
    actions.add(
        cfg.data_dir,
        kind=alert.get("category") or record["kind"],
        summary=alert["suggestion_text"],
        target_match_id=(alert.get("match_ids") or [None])[0],
        details={
            "alert_title": alert.get("title"),
            "ops": record["ops"],
            "suggestion": alert.get("suggestion"),
        },
    )
    return True


def actionable_suggestions(limit: int) -> list[dict]:
    return [a for a in report.all_alerts if a.get("suggestion_text")][:limit]


def critical_alerts(limit: int) -> list[dict]:
    pool = (
        report.field_conflicts
        + report.referee_conflicts
        + [a for a in report.weather_alerts if a["severity"] == "critical"]
        + [a for a in report.roster_shortages if a["severity"] == "critical"]
        + report.weather_alerts
        + report.roster_shortages
    )
    seen, out = set(), []
    for a in pool:
        key = a["title"]
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= limit:
            break
    return out


def recent_notifications(limit: int) -> list[dict]:
    rows: list[dict] = []
    for c in cancel.load(cfg.data_dir)[-limit * 2:]:
        rows.append({
            "icon": "🚫",
            "text": f"Cancellation sent for {c.match_id}",
            "ts": c.timestamp,
        })
    for a in actions.recent(cfg.data_dir, limit=limit * 2):
        rows.append({
            "icon": "✅",
            "text": a.summary,
            "ts": a.timestamp,
        })
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows[:limit]


# ======================================================================
# Page: Dashboard
# ======================================================================
def page_dashboard() -> None:
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:8px;">
          <h1 style="margin-bottom:0">League Management Dashboard</h1>
          <p style="color:#666;margin-top:4px">
            Last updated by AI: {time_ago(st.session_state.diagnostics_run_at)}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")

    # ---- LEFT COLUMN ----
    with left:
        # Upcoming Schedule
        with st.container(border=True):
            card_header("Upcoming Schedule", "View Full Schedule →", "schedule", st)
            ms = report.upcoming_matches[:4]
            if not ms:
                st.caption("No matches scheduled in the next 7 days.")
            else:
                rows = [{
                    "Date": fmt_date(m["date"]),
                    "Time": m["time_start"],
                    "Match": f"{m['home_team_name']} vs {m['away_team_name']}",
                    "Field": m["field_name"],
                    "Referee": m["ref_name"],
                } for m in ms]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Recent Notifications
        with st.container(border=True):
            card_header("Recent Notifications", "View All →", "notifications", st)
            notes = recent_notifications(4)
            if not notes:
                st.caption("No notifications yet. Confirm a suggestion to see activity here.")
            else:
                for n in notes:
                    st.markdown(f"- {n['icon']} {n['text']}")

    # ---- RIGHT COLUMN ----
    with right:
        # Conflict Alerts
        with st.container(border=True):
            card_header("Conflict Alerts", "View All Alerts →", "alerts", st)
            top = critical_alerts(3)
            if not top:
                st.caption("No active conflicts. Schedule is clean.")
            else:
                for a in top:
                    sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                        a["severity"], "⚪"
                    )
                    st.markdown(f"{sev_icon} {a['title']}")

        # AI Suggestions
        with st.container(border=True):
            card_header("AI Suggestions", "View All Suggestions →", "suggestions", st)
            sugs = actionable_suggestions(3)
            if not sugs:
                st.caption("No suggestions to confirm right now.")
            else:
                for i, a in enumerate(sugs):
                    cols = st.columns([5, 1])
                    cols[0].markdown(f"• {a['suggestion_text']}")
                    if cols[1].button("Confirm", key=f"dash_confirm_{i}",
                                      type="primary", use_container_width=True):
                        if confirm_suggestion(a):
                            st.toast("Schedule updated", icon="✅")
                            st.rerun()

        # Weather Forecast strip
        with st.container(border=True):
            card_header("Weather Forecast", "View Full Forecast →", "weather", st)
            location = st.session_state.loc
            today_dt = datetime.strptime(today, "%Y-%m-%d")
            end_dt = today_dt + pd.Timedelta(days=14)
            upcoming = data.matches[
                (data.matches["StartDT"] >= today_dt) & (data.matches["StartDT"] <= end_dt)
            ]
            unique_dates = sorted(upcoming["Date"].unique())[:5]
            if not unique_dates:
                st.caption("No upcoming match days.")
            else:
                cols = st.columns(len(unique_dates))
                for col, d in zip(cols, unique_dates):
                    f = agent.weather.get_forecast(
                        d, lat=location["lat"], lon=location["lon"], city=location["city"]
                    )
                    severe_color = "#9b1c1c" if f.is_severe else "#1f2937"
                    col.markdown(
                        f"""
                        <div style="text-align:center;padding:6px 4px;">
                          <div style="font-size:0.85em;color:#666">{fmt_date(d)}</div>
                          <div style="font-size:1.8em;line-height:1.4">{f.icon}</div>
                          <div style="font-weight:700;color:{severe_color}">{f.temp_f}°F</div>
                          <div style="font-size:0.85em;color:#444">{f.condition}</div>
                          <div style="font-size:0.72em;color:#888">💧{f.precip_chance}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ======================================================================
# Page: Schedule (full)
# ======================================================================
def page_schedule() -> None:
    st.title("📅 Full Schedule")
    st.caption(f"All upcoming matches from {today} onward.")

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    upcoming = data.matches[data.matches["StartDT"] >= today_dt].sort_values("StartDT")
    if upcoming.empty:
        st.info("No upcoming matches.")
        return

    rows = []
    for _, m in upcoming.iterrows():
        rows.append({
            "Match": m["MatchID"] + (" 🚫" if m["MatchID"] in cancelled_ids else ""),
            "Date": fmt_date(m["Date"]),
            "Time": f"{m['TimeStart']}–{m['TimeEnd']}",
            "Home": data.team_name(m["HomeTeamID"]),
            "Away": data.team_name(m["AwayTeamID"]),
            "Field": data.field_name(m["FieldID"]),
            "Referee": data.referee_name(m["RefID"]),
            "H Att": int(m["HomePresent"]),
            "A Att": int(m["AwayPresent"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("👥 Team rosters"):
        team_options = sorted(data.teams.keys())
        sel = st.selectbox("Team", options=team_options,
                           format_func=lambda t: f"{t} — {data.team_name(t)}")
        roster = agent.tools["get_team_roster"](team_id=sel)
        c1, c2 = st.columns(2)
        c1.markdown(f"**Starters ({len(roster['starters'])})**")
        for p in roster["starters"]:
            c1.write(f"- {p}")
        c2.markdown(f"**Standby ({len(roster['standby'])})**")
        for p in roster["standby"]:
            c2.write(f"- {p}")

    with st.expander("🏆 League standings"):
        srows = standings_table(data, today)
        st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
        if srows:
            leader = srows[0]
            st.success(
                f"🏆 Leader: **{leader['Team']}** — "
                f"{leader['W']}W-{leader['L']}L-{leader['T']}T ({leader['PTS']} pts)"
            )


# ======================================================================
# Page: Alerts
# ======================================================================
def page_alerts() -> None:
    st.title("⚠️ Alerts")
    st.caption(
        "Field conflicts, referee conflicts, roster shortages, weather, and "
        "league-rule violations. Each alert links to a specific resolution."
    )

    rematch_label = (
        f"Rematch cap (≥{st.session_state.max_rematches} same-pair games)"
        if st.session_state.get("max_rematches", 0) > 0
        else "Rematch cap (rule disabled — set in sidebar)"
    )
    weekly_label = (
        f"Weekly game limit (≥{st.session_state.max_weekly} games/week)"
        if st.session_state.get("max_weekly", 0) > 0
        else "Weekly game limit (rule disabled — set in sidebar)"
    )
    sections = [
        ("Field conflicts", report.field_conflicts),
        ("Referee conflicts", report.referee_conflicts),
        ("Weather alerts", report.weather_alerts),
        ("Roster shortages", report.roster_shortages),
        (rematch_label, report.rematch_violations),
        (weekly_label, report.weekly_limit_violations),
    ]
    any_shown = False
    for title, items in sections:
        st.subheader(f"{title} ({len(items)})")
        if not items:
            st.success(f"No {title.lower()}.")
            continue
        any_shown = True
        for a in items:
            render_alert_pretty(a)
    if not any_shown:
        st.success("No active alerts. Schedule is clean.")


# ======================================================================
# Page: Suggestions (with confirm buttons)
# ======================================================================
def page_suggestions() -> None:
    st.title("✨ AI Suggestions")
    st.caption(
        "Every alert ships with a specific, UI-ingestable resolution. "
        "Click **Confirm** to apply — the action is logged in Notifications."
    )

    sugs = actionable_suggestions(50)
    if not sugs:
        st.success("No suggestions outstanding.")
        return

    by_kind: dict[str, list[dict]] = {}
    for a in sugs:
        kind = (a.get("suggestion") or {}).get("type") or a["category"]
        by_kind.setdefault(kind, []).append(a)

    for kind, group in by_kind.items():
        st.subheader(f"{kind.replace('_', ' ').title()} ({len(group)})")
        for i, a in enumerate(group):
            with st.container(border=True):
                cols = st.columns([5, 1])
                sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    a["severity"], "⚪"
                )
                cols[0].markdown(
                    f"{sev_icon} **{a['title']}**  \n{a['suggestion_text']}"
                )
                if cols[1].button("Confirm", key=f"sug_confirm_{kind}_{i}",
                                  type="primary", use_container_width=True):
                    if confirm_suggestion(a):
                        st.toast("Schedule updated", icon="✅")
                        st.rerun()
                with st.expander("Structured payload (UI-ingestable)"):
                    st.json(a["suggestion"])


# ======================================================================
# Page: Notifications
# ======================================================================
def page_notifications() -> None:
    st.title("📬 Notifications")
    st.caption(
        "Cancellations and confirmed suggestions, newest first. "
        "These are persisted to `data/cancellations.json` and `data/actions.json`."
    )

    cans = cancel.load(cfg.data_dir)
    acts = actions.load(cfg.data_dir)
    combined: list[tuple[str, str, str, dict]] = []
    for c in cans:
        combined.append((c.timestamp, "🚫 Cancellation",
                         f"{c.match_id} — {c.reason} (notified: {', '.join(c.notified)})",
                         {"kind": "cancellation", "match_id": c.match_id}))
    for a in acts:
        combined.append((a.timestamp, "✅ Confirmed",
                         f"[{a.kind}] {a.summary}",
                         {"kind": "action", "details": a.details}))
    combined.sort(key=lambda x: x[0], reverse=True)

    if not combined:
        st.info("No notifications yet. Confirm a suggestion or send a cancellation to populate.")
        return

    for ts, label, summary, meta in combined:
        with st.container(border=True):
            cols = st.columns([5, 1])
            cols[0].markdown(f"**{label}** · _{ts}_  \n{summary}")
            if meta["kind"] == "cancellation":
                if cols[1].button("Undo", key=f"undo_can_{meta['match_id']}_{ts}"):
                    cancel.remove(cfg.data_dir, meta["match_id"])
                    st.rerun()
            with st.expander("Details"):
                st.json(meta.get("details") or meta)


# ======================================================================
# Page: Weather
# ======================================================================
def page_weather() -> None:
    st.title("🌤️ Weather Forecast")
    st.caption(
        f"Provider: **{agent.weather.name}** · Location: "
        f"**{st.session_state.loc['city']}** "
        f"({st.session_state.loc['lat']:.3f}, {st.session_state.loc['lon']:.3f})"
    )

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    end_dt = today_dt + pd.Timedelta(days=14)
    upcoming = data.matches[
        (data.matches["StartDT"] >= today_dt) & (data.matches["StartDT"] <= end_dt)
    ]
    unique_dates = sorted(upcoming["Date"].unique())
    if not unique_dates:
        st.info("No upcoming match days.")
        return

    location = st.session_state.loc
    for d in unique_dates:
        day_matches = upcoming[upcoming["Date"] == d].sort_values("StartDT")
        f = agent.weather.get_forecast(
            d, lat=location["lat"], lon=location["lon"], city=location["city"]
        )
        bg = "#FFE5E5" if f.is_severe else "#F0FAF0"
        fg = "#9b1c1c" if f.is_severe else "#166534"
        badge = "🚨 SEVERE" if f.is_severe else "✅ Playable"
        st.markdown(
            f"""
            <div style="background:{bg};color:{fg};padding:14px 16px;
                        border-radius:10px;margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-size:1.4em">{f.icon}</span>
                  <b style="font-size:1.05em">&nbsp;{fmt_date(d)}</b> &nbsp;·&nbsp;
                  {f.condition} &nbsp;·&nbsp; {f.temp_f}°F &nbsp;·&nbsp;
                  💧 {f.precip_chance}% &nbsp;·&nbsp; 💨 {f.wind_mph} mph
                </div>
                <div><b>{badge}</b></div>
              </div>
              <div style="margin-top:6px;font-size:0.92em;">
                {f.advisory} <i>(source: {f.source})</i>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for _, m in day_matches.iterrows():
            mid = m["MatchID"]
            is_cancelled = mid in cancelled_ids
            cols = st.columns([5, 2])
            label = (
                f"~~{mid}~~ — ~~{data.team_name(m['HomeTeamID'])} vs "
                f"{data.team_name(m['AwayTeamID'])}~~ 🚫 cancelled"
            ) if is_cancelled else (
                f"**{mid}** — {m['TimeStart']} · "
                f"{data.team_name(m['HomeTeamID'])} vs "
                f"{data.team_name(m['AwayTeamID'])} · "
                f"{data.field_name(m['FieldID'])}"
            )
            cols[0].markdown(label)
            if f.is_severe:
                if is_cancelled:
                    if cols[1].button("Restore", key=f"wx_restore_{mid}"):
                        cancel.remove(cfg.data_dir, mid)
                        st.rerun()
                else:
                    if cols[1].button("📢 Send Cancellation", key=f"wx_cancel_{mid}",
                                      type="primary"):
                        cancel.add(
                            cfg.data_dir, mid,
                            reason=(
                                f"{f.condition} forecast — {f.precip_chance}% precip, "
                                f"{f.wind_mph} mph wind"
                            ),
                            forecast_snapshot={
                                "date": f.date, "location": f.location,
                                "condition": f.condition, "temp_f": f.temp_f,
                                "precip_chance": f.precip_chance, "wind_mph": f.wind_mph,
                                "source": f.source,
                            },
                        )
                        actions.add(
                            cfg.data_dir, kind="weather_cancellation",
                            summary=f"Cancelled {mid} due to {f.condition} on {d}",
                            target_match_id=mid,
                        )
                        st.rerun()

    if agent.weather.name == "mock" and FORCED_BAD_WEATHER:
        st.caption(
            "Mock provider · seeded bad-weather dates: "
            + ", ".join(f"{d} ({c[0]})" for d, c in FORCED_BAD_WEATHER.items())
        )


# ======================================================================
# Page: Chat
# ======================================================================
def page_chat() -> None:
    st.title("💬 Agent Chat")
    st.caption(
        f"Active LLM: **{LLM_DISPLAY_NAME}** · Weather: **{agent.weather.name}** · "
        f"Location: **{st.session_state.loc['city']}**."
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    suggested = [
        "Are there any conflicts I should know about?",
        "Do any teams have roster shortages this week?",
        "What's the weather impact on upcoming games?",
        "Show me details on M1377",
        "Who is leading the league right now?",
    ]
    cols = st.columns(len(suggested))
    for i, q in enumerate(suggested):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state.pending_q = q

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("Tool trace"):
                    for t in msg["trace"]:
                        st.write(f"**Step {t['step']}** — `{t['tool']}({t['args']})`")
                        st.code(t["result_preview"], language="json")

    user_q = st.chat_input("Ask the agent…")
    if "pending_q" in st.session_state:
        user_q = user_q or st.session_state.pop("pending_q")

    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking via {LLM_DISPLAY_NAME}…"):
                result = agent.chat(user_q, today)
            st.markdown(result["answer"])
            if result.get("trace"):
                with st.expander("Tool trace"):
                    for t in result["trace"]:
                        st.write(f"**Step {t['step']}** — `{t['tool']}({t['args']})`")
                        st.code(t["result_preview"], language="json")
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["answer"],
            "trace": result.get("trace", []),
        })


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------
PAGE_FNS = {
    "dashboard": page_dashboard,
    "schedule": page_schedule,
    "alerts": page_alerts,
    "suggestions": page_suggestions,
    "notifications": page_notifications,
    "weather": page_weather,
    "chat": page_chat,
}
PAGE_FNS.get(st.session_state.page, page_dashboard)()
