# settings.py
import os
from pathlib import Path

class Settings:
    # 基础配置
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    MEMOS_DIR: Path = Path(os.getcwd())
    # 日志 API 配置 (你的 Loguru Sink 会用到)
    CUSTOM_LOGGER_URL: str = os.getenv("CUSTOM_LOGGER_URL", "")
    CUSTOM_LOGGER_TOKEN: str = os.getenv("CUSTOM_LOGGER_TOKEN", "")
    LOG_FILTER_TREE_PREFIX: str = ''
settings = Settings()