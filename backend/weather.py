"""Weather forecasting.

Three providers behind a single `WeatherProvider.get_forecast()` interface:

  1. `openweather` — OpenWeatherMap 5-day / 3-hour forecast
                     (https://openweathermap.org/forecast5). Requires an API
                     key. Includes geocoding via OWM Geocoding API.

  2. `open-meteo`  — Open-Meteo daily forecast (https://open-meteo.com/).
                     No API key required, free for non-commercial use,
                     up to 16 days of daily forecasts. Recommended for
                     longer planning horizons.

  3. `mock`        — Deterministic, dependency-free fallback used for the demo
                     and seeded with `FORCED_BAD_WEATHER` dates so the severe-
                     weather UI flow is always visible.

For dates outside a real provider's forecast horizon, providers fall back to
`mock` and label the result `source="seasonal estimate"` so the UI can be
honest about what it's showing.
"""
from __future__ import annotations
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# ----------------------------------------------------------------------
# Demo seed — guarantees visible severe-weather alerts in mock mode.
# ----------------------------------------------------------------------
FORCED_BAD_WEATHER = {
    "2026-04-28": ("Thunderstorm", 92, 28),  # severe
    "2026-04-30": ("Rain", 80, 18),           # rain delay
    "2026-05-02": ("Drizzle", 55, 12),        # borderline
}

SEVERE_CONDITIONS = {"Thunderstorm", "Tornado", "Hurricane", "Snow"}
CONDITION_RANK = {
    "Tornado": 6, "Hurricane": 6, "Thunderstorm": 5,
    "Snow": 4, "Rain": 3, "Drizzle": 2,
    "Mist": 1, "Fog": 1, "Haze": 1, "Smoke": 1,
    "Clouds": 1, "Clear": 0,
}
CONDITION_ICON = {
    "Thunderstorm": "⛈️", "Tornado": "🌪️", "Hurricane": "🌀",
    "Rain": "🌧️", "Drizzle": "🌦️", "Snow": "❄️",
    "Clouds": "☁️", "Clear": "☀️", "Mist": "🌫️",
    "Fog": "🌫️", "Haze": "🌫️", "Partly Cloudy": "⛅",
}


@dataclass
class WeatherForecast:
    date: str
    location: str
    condition: str
    temp_f: int
    precip_chance: int
    wind_mph: int
    is_playable: bool
    advisory: str
    source: str  # "openweather" | "open-meteo" | "mock" | "seasonal estimate" | "mock (api unavailable)"

    @property
    def is_severe(self) -> bool:
        return not self.is_playable

    @property
    def icon(self) -> str:
        return CONDITION_ICON.get(self.condition, "🌡️")


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _hash01(*parts: str) -> float:
    h = hashlib.md5("|".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") / 0xFFFFFFFF


def _classify(condition: str, precip: int) -> tuple[bool, str]:
    if condition in SEVERE_CONDITIONS or precip >= 70:
        return False, f"{condition} forecast — reschedule recommended."
    if precip >= 50:
        return True, f"{condition} possible — monitor field conditions."
    return True, "Conditions look good for play."


def _mock_forecast(date: str, location: str, *, source: str = "mock") -> WeatherForecast:
    if date in FORCED_BAD_WEATHER:
        condition, precip, wind = FORCED_BAD_WEATHER[date]
        temp = 60 + int(_hash01(date, location, "t") * 10)
        playable, advisory = _classify(condition, precip)
        return WeatherForecast(date, location, condition, temp, precip, wind,
                               playable, advisory, source=source)
    r = _hash01(date, location)
    if r < 0.65:
        condition, precip = "Clear", int(_hash01(date, location, "p") * 10)
    elif r < 0.85:
        condition, precip = "Clouds", 10 + int(_hash01(date, location, "p") * 20)
    elif r < 0.95:
        condition, precip = "Drizzle", 30 + int(_hash01(date, location, "p") * 25)
    else:
        condition, precip = "Rain", 60 + int(_hash01(date, location, "p") * 25)
    temp = 62 + int(_hash01(date, location, "t") * 22)
    wind = 4 + int(_hash01(date, location, "w") * 14)
    playable, advisory = _classify(condition, precip)
    return WeatherForecast(date, location, condition, temp, precip, wind,
                           playable, advisory, source=source)


# ----------------------------------------------------------------------
# Provider abstraction
# ----------------------------------------------------------------------
class WeatherProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def get_forecast(
        self, date: str, *, lat: Optional[float] = None,
        lon: Optional[float] = None, city: Optional[str] = None,
    ) -> WeatherForecast: ...

    def geocode(self, query: str) -> Optional[tuple[float, float, str]]:
        """Optional. Returns (lat, lon, formatted_name) or None."""
        return None


class MockWeatherProvider(WeatherProvider):
    name = "mock"

    def get_forecast(self, date, *, lat=None, lon=None, city=None):
        return _mock_forecast(date, city or "Demo City")


class OpenWeatherProvider(WeatherProvider):
    """OpenWeatherMap 5-day / 3-hour forecast. Free tier."""
    name = "openweather"
    BASE = "https://api.openweathermap.org"

    def __init__(self, api_key: str, cache_ttl: int = 600):
        self.api_key = api_key
        self._cache: dict[tuple[float, float], tuple[float, dict]] = {}
        self._geocode_cache: dict[str, tuple[float, float, str]] = {}
        self._ttl = cache_ttl

    def geocode(self, query: str):
        if not query:
            return None
        if query in self._geocode_cache:
            return self._geocode_cache[query]
        import requests
        try:
            r = requests.get(
                f"{self.BASE}/geo/1.0/direct",
                params={"q": query, "limit": 1, "appid": self.api_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        if not data:
            return None
        d = data[0]
        country = d.get("country", "")
        state = d.get("state", "")
        bits = [d["name"], state, country]
        formatted = ", ".join(b for b in bits if b)
        result = (float(d["lat"]), float(d["lon"]), formatted)
        self._geocode_cache[query] = result
        return result

    def _fetch(self, lat: float, lon: float) -> dict:
        key = (round(lat, 2), round(lon, 2))
        cached = self._cache.get(key)
        now = time.time()
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        import requests
        r = requests.get(
            f"{self.BASE}/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "units": "imperial", "appid": self.api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        self._cache[key] = (now, data)
        return data

    def get_forecast(self, date, *, lat=None, lon=None, city=None):
        location = city or (f"{lat:.2f}, {lon:.2f}" if lat is not None else "Unknown")
        if lat is None or lon is None:
            return _mock_forecast(date, location)
        try:
            data = self._fetch(lat, lon)
        except Exception:
            return _mock_forecast(date, location, source="mock (api unavailable)")
        entries = [e for e in data.get("list", []) if e.get("dt_txt", "").startswith(date)]
        if not entries:
            return _mock_forecast(date, location, source="seasonal estimate")
        max_pop = max(float(e.get("pop", 0)) for e in entries)
        max_wind = max(float(e["wind"]["speed"]) for e in entries)
        midday = min(entries, key=lambda e: abs(int(e["dt_txt"][11:13]) - 13))
        temp_f = int(round(float(midday["main"]["temp"])))
        worst = max(entries, key=lambda e: CONDITION_RANK.get(e["weather"][0]["main"], 0))
        condition = worst["weather"][0]["main"]
        precip_pct = int(round(max_pop * 100))
        playable, advisory = _classify(condition, precip_pct)
        return WeatherForecast(date, location, condition, temp_f, precip_pct,
                               int(round(max_wind)), playable, advisory,
                               source="openweather")


class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo daily forecast. No API key, up to 16 days."""
    name = "open-meteo"
    BASE = "https://api.open-meteo.com/v1/forecast"
    GEO_BASE = "https://geocoding-api.open-meteo.com/v1/search"

    # WMO weather codes -> our condition vocabulary
    # https://open-meteo.com/en/docs (WMO Weather interpretation codes)
    CODE_MAP = {
        0: "Clear", 1: "Clear", 2: "Clouds", 3: "Clouds",
        45: "Fog", 48: "Fog",
        51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
        56: "Drizzle", 57: "Drizzle",
        61: "Rain", 63: "Rain", 65: "Rain",
        66: "Rain", 67: "Rain",
        71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
        80: "Rain", 81: "Rain", 82: "Rain",
        85: "Snow", 86: "Snow",
        95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
    }

    def __init__(self, cache_ttl: int = 600):
        self._cache: dict[tuple[float, float], tuple[float, dict]] = {}
        self._geocode_cache: dict[str, tuple[float, float, str]] = {}
        self._ttl = cache_ttl

    def geocode(self, query: str):
        if not query:
            return None
        if query in self._geocode_cache:
            return self._geocode_cache[query]
        import requests
        try:
            r = requests.get(self.GEO_BASE, params={"name": query, "count": 1},
                             timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        results = data.get("results") or []
        if not results:
            return None
        d = results[0]
        bits = [d["name"], d.get("admin1") or "", d.get("country_code") or ""]
        formatted = ", ".join(b for b in bits if b)
        result = (float(d["latitude"]), float(d["longitude"]), formatted)
        self._geocode_cache[query] = result
        return result

    def _fetch(self, lat: float, lon: float) -> dict:
        key = (round(lat, 2), round(lon, 2))
        cached = self._cache.get(key)
        now = time.time()
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        import requests
        r = requests.get(
            self.BASE,
            params={
                "latitude": lat, "longitude": lon,
                "daily": ",".join([
                    "weather_code", "temperature_2m_max",
                    "precipitation_probability_max", "wind_speed_10m_max",
                ]),
                "timezone": "auto",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "forecast_days": 16,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        self._cache[key] = (now, data)
        return data

    def get_forecast(self, date, *, lat=None, lon=None, city=None):
        location = city or (f"{lat:.2f}, {lon:.2f}" if lat is not None else "Unknown")
        if lat is None or lon is None:
            return _mock_forecast(date, location)
        try:
            data = self._fetch(lat, lon)
        except Exception:
            return _mock_forecast(date, location, source="mock (api unavailable)")
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        if date not in dates:
            return _mock_forecast(date, location, source="seasonal estimate")
        idx = dates.index(date)
        code = int((daily.get("weather_code") or [0])[idx])
        condition = self.CODE_MAP.get(code, "Clouds")
        temp_f = int(round(float((daily.get("temperature_2m_max") or [70])[idx])))
        precip_pct = int(round(float((daily.get("precipitation_probability_max") or [0])[idx] or 0)))
        wind_mph = int(round(float((daily.get("wind_speed_10m_max") or [5])[idx])))
        playable, advisory = _classify(condition, precip_pct)
        return WeatherForecast(date, location, condition, temp_f, precip_pct,
                               wind_mph, playable, advisory, source="open-meteo")


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def get_provider(cfg) -> WeatherProvider:
    choice = (cfg.weather_provider or "mock").lower()
    if choice == "openweather":
        if not cfg.is_openweather_ready():
            return MockWeatherProvider()
        return OpenWeatherProvider(cfg.openweather_api_key)
    if choice == "open-meteo":
        return OpenMeteoProvider()
    return MockWeatherProvider()
