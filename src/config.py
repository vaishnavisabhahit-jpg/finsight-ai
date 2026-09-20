from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    DATA_PATH: str = "data/sec_features.csv"
    ANALYTICS_PATH: str = "outputs/sec_master_analytics.csv"
    ALLOWED_ORIGINS: str = "http://localhost:5173,https://finsight-ai.vercel.app"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()