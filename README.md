# Little League Manager — Agentic Scheduling MVP

An end-to-end **single-agent** system that monitors a youth sports league for
real-time scheduling problems and produces **specific, UI-ingestable
resolutions** (Field #, Time, Referee, Cancellation).

Built for the IBM Experiential AI Learning Lab. Two pluggable backends:

| Backend | Choices |
|---|---|
| **LLM** | `watsonx` (IBM Cloud) · `custom` (any OpenAI-compatible endpoint — vLLM / Ollama / LM Studio) · `mock` |
| **Weather** | `openweather` (5-day key) · `open-meteo` (16-day, no key) · `mock` |

Both work with **zero API keys** out of the box via the mock providers.

```
              ┌─────────────────┐        ┌─────────────────────────────┐
              │  League CSVs    │ ─────▶ │  Data Loader (pandas)       │
              │  (5 sources)    │        │  matches/fields/refs/etc.   │
              └─────────────────┘        └──────────────┬──────────────┘
                                                        │
                              ┌─────────────────────────┴──────────────────────┐
                              ▼                                                ▼
              ┌────────────────────────────┐               ┌──────────────────────────────┐
              │ Autonomous Diagnostics     │               │ LLM Provider                 │
              │ (rules-based, always-on)   │               │  • watsonx (Granite, Llama)  │
              │                            │               │  • custom (vLLM/Ollama/…)    │
              │ • Field conflicts          │               │  • mock (no API)             │
              │ • Referee conflicts        │               └─────────────┬────────────────┘
              │ • Roster (<9 / tight 9)    │                             │
              │ • Rematch / weekly limits  │                             ▼
              │ • Home/away balance        │               ┌──────────────────────────────┐
              │ • Weather forecast         │               │ Tool-using Agent (ReAct)     │
              └─────────────┬──────────────┘               └─────────────┬────────────────┘
                            │                                            │
                            ▼                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Structured Alerts + Specific Suggestions               │
                  │ (Field #, Time, Referee — UI-ingestable JSON)          │
                  └─────────────────────────┬──────────────────────────────┘
                                            ▼
                              ┌──────────────────────────┐
                              │ Streamlit UI             │
                              │ Dashboard / Conflicts /  │
                              │ Roster / Weather /       │
                              │ Standings / Chat         │
                              └──────────────────────────┘
```

## Project layout

```
LITTLE_LEAGUE_MANAGER/
├── .env / .env.example       # provider configuration
├── data/                     # 5 CSVs (referees, fields, matches, schedule, roster)
├── backend/
│   ├── config.py             # loads .env
│   ├── data_loader.py        # pandas-backed league dataset
│   ├── weather.py            # mock / openweather / open-meteo providers
│   ├── cancellations.py      # persistent "Send Cancellation" audit log
│   ├── standings.py          # win/loss/forfeit calculation
│   ├── tools.py              # ALL agent tools (lookups + validators + suggestions)
│   ├── llm_provider.py       # watsonx / custom / mock abstraction
│   ├── agent.py              # diagnostics pass + ReAct chat loop
│   └── smoketest.py          # CLI sanity check
├── frontend/app.py           # Streamlit UI (6 tabs)
├── requirements.txt
├── run.sh                    # ./run.sh install | app | test
└── README.md
```

## Quick start

```bash
# 1. Install
./run.sh install
# or:  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Smoke-test the backend (no LLM required)
./run.sh test

# 3. Launch the Streamlit UI
./run.sh app
# opens http://localhost:8501
```

The default `.env` ships with `LLM_PROVIDER=mock` so the app boots immediately
with no credentials. Switch providers via the sidebar dropdown or by editing
`.env`.

## Switching the LLM backend

Edit `.env` (or use the sidebar):

```bash
# Option A — IBM watsonx.ai (cloud)
LLM_PROVIDER=watsonx
WATSONX_APIKEY=<your IAM api key>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=<your project id>
WATSONX_MODEL_ID=ibm/granite-3-8b-instruct          # or meta-llama/llama-3-3-70b-instruct, etc.

# Option B — any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, …)
LLM_PROVIDER=custom
CUSTOM_LLM_BASE_URL=http://localhost:8080/v1
CUSTOM_LLM_API_KEY=not-needed                       # or your key
CUSTOM_LLM_MODEL=llama-3.1-8b-instruct

# Option C — mock (no external calls; deterministic responses)
LLM_PROVIDER=mock
```

The agent loop is identical for all three — they share the
`LLMProvider.complete(system, user)` interface in `backend/llm_provider.py`.

## Switching the weather backend

```bash
# Option A — Open-Meteo (RECOMMENDED for this app: free, no key, 16-day horizon)
WEATHER_PROVIDER=open-meteo

# Option B — OpenWeatherMap (5-day forecast, free key from
#            https://home.openweathermap.org/users/sign_in)
WEATHER_PROVIDER=openweather
OPENWEATHER_API_KEY=<your key>

# Option C — mock (deterministic seeded weather, demo-friendly)
WEATHER_PROVIDER=mock

# League location (overridable from the UI sidebar; geocoded by both real providers)
LEAGUE_CITY=Austin, US
LEAGUE_LAT=30.27
LEAGUE_LON=-97.74
```

The provider can also be switched at runtime from the sidebar dropdown. The
sidebar's **📍 League location** form lets you type any city, click
**🔍 Geocode** to resolve it to lat/lon, then **Apply** to push the new location
into the agent. All weather panels and the diagnostics pass re-render against
the live location.

> Why Open-Meteo by default? OpenWeather's free tier is capped at 5 forecast
> days, but the dataset's match schedule extends weeks ahead. Open-Meteo
> returns 16 days, no key, no rate-limit fuss. OpenWeather is supported per
> spec; Open-Meteo is the better fit for league planning. Either way, dates
> outside the provider's horizon fall back to the mock provider with a
> visible `source: seasonal estimate` label so the UI is honest about what
> it's showing.

## What the agent does

### Autonomous diagnostics (no LLM required)
On every dashboard load the agent runs:

| Check | Severity | Suggestion shape |
|---|---|---|
| **Field double-booking** | critical | `{type: field_reassignment, match_id, new_field_id, fallback_slot}` |
| **Referee double-booking** | critical | `{type: referee_reassignment, match_id, new_ref_id, new_ref_name}` |
| **Roster shortage** (<9 players) | warning / critical | `{type: roster_callup, match_id, team_id, call_ups: […]}` |
| **Tight roster** (exactly 9) | info | same shape — recommends 1 buffer player |
| **Weather alert** (rain/storm) | warning / critical | `{type: weather_reschedule, match_id, alternative: {date,time,field_id,ref_id}, forecast: {…}}` |
| **Rematch cap** (>2 between same pair) | warning | `{type: cancel_rematch, match_id, pair}` |
| **Weekly limit** (>2 games / 7 days) | warning | `{type: defer_match, match_ids, team_id}` |
| **Home/away imbalance** (>±2) | info | `{type: swap_home_away, team_id, direction}` |

Every alert carries a **specific, structured `suggestion`** — exactly what the
problem statement asks for. The Streamlit UI renders the human-readable
`suggestion_text` and exposes the raw JSON in an expander labeled "UI-ingestable
JSON," so a real frontend can wire an "Apply" button straight to it.

### ReAct chat (LLM-driven)
For free-form questions the agent runs a JSON tool-calling loop using the
configured LLM. Tools:

- `get_upcoming_matches`, `get_match_details`, `get_team_roster`,
  `get_standby_players`, `get_available_fields`, `get_available_referees`
- All of the validators above
- `suggest_reschedule(match_id, today)` → date / time / field / ref triple

Falls back to a deterministic responder when `LLM_PROVIDER=mock`, so the chat
tab works in every configuration.

## Standings

Computed from completed matches (`date < today`) using:

* both teams < 9 players  → double forfeit
* one team < 9 players    → that team forfeits
* both teams ≥ 9 players  → higher attendance wins; equal attendance is a tie

Standings are sorted by `pts (3*W + 1*T) → wins → win % → name`.

## Send Weather Cancellation

When the dashboard's **Weather outlook** flags a severe day, every match on that
day gets a **📢 Send Weather Cancellation** button. Clicking it:

1. Appends a record to [`data/cancellations.json`](data/cancellations.json) with
   the timestamp, match ID, reason (e.g., `"Thunderstorm forecast — 92% precip,
   28 mph wind"`), the list of recipients (`Home Coach, Away Coach, Assigned
   Referee, League Coordinator`), and a snapshot of the forecast.
2. Visually marks the match as cancelled (strikethrough) on the dashboard and
   in the upcoming matches table (🚫 indicator).
3. Surfaces an undoable entry in the **Cancellation log** on the Weather tab.

The audit log is the system's source of truth — a future iteration can wire it
to email / SMS / push without touching the UI.

## Reproducible demo dates

Seeded "bad weather" dates so the weather-driven reschedule flow always lights
up on the dashboard:

* **2026-04-28** — Thunderstorms (92% precip) → all 6 matches re-routed
* **2026-04-30** — Heavy Rain (80%)
* **2026-05-02** — Light Rain (55%)

Pin `DEMO_TODAY=2026-04-25` in `.env` (default) for reproducible demos.

## Data sources

All five CSVs live under [`data/`](data/):

| File | Rows | Schema |
|---|---|---|
| `roster_final.csv` | 152 | `TeamID, TeamName, PlayerName` |
| `league_schedule_final.csv` | 199 | `Date, TimeStart, TimeEnd` |
| `matches_final.csv` | 411 | `MatchID, Date, TimeStart, TimeEnd, Home/Away TeamID, FieldID, RefID, Home/AwayPresent` |
| `fields_final.csv` | 411 | `FieldID, FieldName, Date, BookedStart, BookedEnd, MatchID` |
| `referees_final.csv` | 512 | `RefID, RefereeName, Date, AvailabilityStart, AvailabilityEnd` |

12 teams, 3 fields, 8 referees, dataset spans 2026-01-15 → 2026-05-05.

## Extending

* **New validation rule** — add a function to `backend/tools.py` (returns the
  same alert dict shape) and wire it into `LeagueManagerAgent.run_diagnostics`
  in `backend/agent.py`.
* **Real weather** — replace `backend/weather.py:get_forecast` with a call to
  any weather API. Keep the `WeatherForecast` dataclass shape.
* **Persistence** — currently CSV-only. Swap `backend/data_loader.py` for a
  database loader; the rest of the stack is dataframe-agnostic.
