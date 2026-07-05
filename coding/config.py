import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="env")

class Settings:


    DATABASE_URL: str = os.getenv("DATABASE_URL")

    MODEL_URI: str = os.getenv("MODEL_URI")
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

settings = Settings()
