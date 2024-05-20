import logging
import logging.handlers
import os
from functools import lru_cache

import ecs_logging
import openai
from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings class for application settings and secrets management.

    Official documentation on pydantic settings management.

    - https://pydantic-docs.helpmanual.io/usage/settings/.
    """

    # Application Path
    APP_NAME: str = "Intella"
    REPO_PATH: str = os.path.abspath(".")
    DATALINK_PATH: str = os.path.join(REPO_PATH, "datalink")
    SETTINGS_PATH: str = os.path.join(REPO_PATH, "settings")
    ENV_PATH: str = os.path.join(REPO_PATH, ".env")

    # Logger
    LOG_VERBOSITY: str = "INFO"
    LOG_ROTATION_SIZE: str = "100MB"
    LOG_RETENTION: str = "30 days"
    LOG_FILE_NAME: str = "./logs/intella_{time:D-M-YY}.log"
    LOG_FORMAT: str = "{time:HH:mm:ss!UTC}\t|\t{file}:{module}:{line}\t|\t{message}"
    ECS_LOG_PATH: str = f"./logs/elastic-{os.getpid()}.log"
    PROFILE: bool = False

    OPENAI_KEY: str = ""

    # Qdrant
    CACHE_PATH: str = "./data/qdrant/cache"
    USE_CACHE: bool = True  # if you want to load the cached parquet files with the embeddings from disk
    USE_VERSIONING: bool = True  # if you want to save on parquet files the dataset with the embeddings
    # Debug mode: use into embedding function to reduce the number of articles to generate (remember to put on false before to deploy)
    DEBUG_MODE: bool = False
    ENV_LOAD: str = ""
    COLLECTION_NAME: str = "test"
    # QDRANT VARIABLES
    QDRANT_HOST: str = "http://localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = "dev"
    QDRANT_COLLECTION_NAME: str = "test"

    # Dataset
    MAX_ARTICLES: int = 10

    def _configure_openai(self, openai_key) -> bool:
        self.OPENAI_KEY = openai_key
        openai.api_key = self.OPENAI_KEY

    def _setup_logger(self) -> bool:
        # logger.remove() to remove default logging to StdErr
        logger.add(
            self.LOG_FILE_NAME,
            rotation=self.LOG_ROTATION_SIZE,
            retention=self.LOG_RETENTION,
            colorize=True,
            format=self.LOG_FORMAT,
            level=self.LOG_VERBOSITY,
            serialize=False,
            catch=True,
            backtrace=False,
            diagnose=False,
            encoding="utf8",
        )

        # Proxy loguru logs also to logging logger.
        # The ecs logging formats all logs from the python logging system for elastic.
        # It could be configured to read logs directly from loguru, but in that case it
        # would miss all parts of the system that log directly to the python logging
        # system.
        class PropagateHandler(logging.Handler):
            def emit(self, record):
                logging.getLogger(record.name).handle(record)

        logger.add(PropagateHandler(), format="{message}")

        # Add ECS rotating files sink
        pylogger = logging.getLogger()
        ecs_handler = logging.handlers.RotatingFileHandler(
            self.ECS_LOG_PATH,
            maxBytes=100_000_000,
            backupCount=2,
            encoding="utf8",
        )
        ecs_handler.setFormatter(
            ecs_logging.StdlibFormatter(
                extra={
                    "release": self.RELEASE,
                    "project": self.PROJECT_NAME,
                }
            )
        )
        ecs_handler.setLevel(logging.INFO)
        pylogger.addHandler(ecs_handler)

        return True


@lru_cache()
def get_settings() -> Settings:
    """Generate and get the settings."""
    try:
        settings = Settings()
        openai.api_key = settings.OPENAI_KEY
        return settings
    except Exception as message:
        logger.error(f"Error: impossible to get the settings: {message}")
        return None


settings = get_settings()
