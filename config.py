"""Configuration management."""
import os
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


class Config:
    """Bot configuration from environment variables."""

    # Telegram Bot
    BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "").strip()
    if not BOT_TOKEN:
        raise SystemExit("Set BOT_TOKEN env var, e.g. export BOT_TOKEN='123:ABC...'")

    # Database
    DB_PATH: str = os.environ.get("DB_PATH", "stats.db")

    # Message time estimation
    TYPING_CHARS_PER_MIN: int = int(os.environ.get("TYPING_CHARS_PER_MIN", "200"))
    SECONDS_OVERHEAD_PER_MSG: int = int(os.environ.get("SECONDS_OVERHEAD_PER_MSG", "2"))

    # FACEIT
    _faceit_key = os.environ.get("FACEIT_API_KEY", "").strip()
    FACEIT_API_KEY: Optional[str] = _faceit_key if _faceit_key else None
    FACEIT_GAME: str = os.environ.get("FACEIT_GAME", "cs2").strip().lower()
    FACEIT_BASE: str = "https://open.faceit.com/data/v4"
    FACEIT_CACHE_TTL_SEC: int = int(os.environ.get("FACEIT_CACHE_TTL_SEC", "300"))
    FACEIT_MAX_CONCURRENCY: int = int(os.environ.get("FACEIT_MAX_CONCURRENCY", "3"))
    
    # OpenAI for advanced summarization (optional)
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY", "").strip() or None
    USE_OPENAI_SUMMARY: bool = os.environ.get("USE_OPENAI_SUMMARY", "false").lower() == "true"
