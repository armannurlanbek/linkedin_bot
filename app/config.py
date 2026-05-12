from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    database_url_sync: str
    tavily_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
