from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    database_url_sync: str
    tavily_api_key: str = ""
    app_password: str = ""
    cookie_secret: str = "change-me-in-production"
    cookie_secure: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
