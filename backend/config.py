"""Loads configuration from environment / .env file."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    llm_provider: str
    watsonx_apikey: str
    watsonx_url: str
    watsonx_project_id: str
    watsonx_model_id: str
    custom_llm_base_url: str
    custom_llm_api_key: str
    custom_llm_model: str
    demo_today: str
    data_dir: Path
    # Weather
    weather_provider: str
    openweather_api_key: str
    league_city: str
    league_lat: float
    league_lon: float

    @classmethod
    def from_env(cls) -> "Config":
        data_dir_raw = os.getenv("DATA_DIR", "data")
        data_dir = (PROJECT_ROOT / data_dir_raw).resolve()
        try:
            lat = float(os.getenv("LEAGUE_LAT", "30.27"))
        except ValueError:
            lat = 30.27
        try:
            lon = float(os.getenv("LEAGUE_LON", "-97.74"))
        except ValueError:
            lon = -97.74
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
            watsonx_apikey=os.getenv("WATSONX_APIKEY", "").strip(),
            watsonx_url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip(),
            watsonx_project_id=os.getenv("WATSONX_PROJECT_ID", "").strip(),
            watsonx_model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct").strip(),
            custom_llm_base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:8080/v1").strip(),
            custom_llm_api_key=os.getenv("CUSTOM_LLM_API_KEY", "not-needed").strip(),
            custom_llm_model=os.getenv("CUSTOM_LLM_MODEL", "llama-3.1-8b-instruct").strip(),
            demo_today=os.getenv("DEMO_TODAY", "").strip(),
            data_dir=data_dir,
            weather_provider=os.getenv("WEATHER_PROVIDER", "mock").strip().lower(),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
            league_city=os.getenv("LEAGUE_CITY", "Austin, US").strip(),
            league_lat=lat,
            league_lon=lon,
        )

    def is_watsonx_ready(self) -> bool:
        return bool(self.watsonx_apikey and self.watsonx_project_id)

    def is_openweather_ready(self) -> bool:
        return bool(self.openweather_api_key)
