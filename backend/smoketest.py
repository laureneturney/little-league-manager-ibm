"""Quick CLI sanity check — no LLM required."""
from __future__ import annotations
import json

from .agent import LeagueManagerAgent
from .config import Config
from .data_loader import load_all
from .llm_provider import get_provider
from .standings import standings_table
from .weather import get_provider as get_weather_provider


def main() -> None:
    cfg = Config.from_env()
    cfg.llm_provider = "mock"  # force deterministic for testing
    data = load_all(cfg.data_dir)
    agent = LeagueManagerAgent(
        data=data,
        llm=get_provider(cfg),
        weather_provider=get_weather_provider(cfg),
        location={"lat": cfg.league_lat, "lon": cfg.league_lon, "city": cfg.league_city},
    )
    today = cfg.demo_today or "2026-04-25"

    report = agent.run_diagnostics(today)
    counts = report.summary_counts()
    print(f"=== Diagnostics for {today} ===")
    print(f"  upcoming(7d): {len(report.upcoming_matches)}")
    print(f"  field conflicts:    {len(report.field_conflicts)}")
    print(f"  referee conflicts:  {len(report.referee_conflicts)}")
    print(f"  roster shortages:   {len(report.roster_shortages)}")
    print(f"  weather alerts:     {len(report.weather_alerts)}")
    print(f"  rematch violations: {len(report.rematch_violations)}")
    print(f"  weekly violations:  {len(report.weekly_limit_violations)}")
    print(f"  totals: {counts}")

    print("\n=== Sample alerts ===")
    for a in report.all_alerts[:5]:
        print(f"  [{a['severity'].upper()}] {a['title']}")
        print(f"    → {a['suggestion_text']}")

    print("\n=== Standings (top 5) ===")
    for row in standings_table(data, today)[:5]:
        print(f"  {row['Team']:30s}  {row['W']}-{row['L']}-{row['T']}  pts={row['PTS']}")

    print("\n=== Mock chat ===")
    res = agent.chat("Are there any conflicts I should know about today?", today)
    print(res["answer"][:800])


if __name__ == "__main__":
    main()
