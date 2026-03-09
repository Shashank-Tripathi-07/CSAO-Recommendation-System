from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    MODEL_PATH: str = "."
    LOG_LEVEL: str = "INFO"
    AWS_REGION: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()
