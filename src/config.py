from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    DATA_PATH: str = str(BASE_DIR / "data" / "synthetic_fpa_features.csv")
    ANALYTICS_PATH: str = str(BASE_DIR / "outputs" / "synthetic_fpa_master_analytics.csv")
    ALLOWED_ORIGINS: str = "http://localhost:5173,https://finsight-ai.vercel.app"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()