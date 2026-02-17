"""Configuration management."""
import os
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip
except Exception as e:
    import logging
    logging.warning(f"Failed to load .env file: {e}")


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
    _openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    OPENAI_API_KEY: Optional[str] = _openai_key if _openai_key else None
    _use_openai = os.environ.get("USE_OPENAI_SUMMARY", "false").strip().lower()
    USE_OPENAI_SUMMARY: bool = _use_openai in ("true", "1", "yes", "on")
    
    # Admin configuration
    ADMIN_USERNAME: str = os.environ.get("ADMIN_USERNAME", "akhmadsadaiev").strip()
    ADMIN_USER_ID: Optional[int] = None
    _admin_user_id_str = os.environ.get("ADMIN_USER_ID", "").strip()
    if _admin_user_id_str:
        try:
            ADMIN_USER_ID = int(_admin_user_id_str)
        except ValueError:
            ADMIN_USER_ID = None
    
    # Log OpenAI config for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"OpenAI config: USE_OPENAI_SUMMARY={USE_OPENAI_SUMMARY}, has_key={bool(OPENAI_API_KEY)}, raw_value='{_use_openai}'")
