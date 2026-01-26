"""FACEIT API client with caching."""
import asyncio
import time
from typing import Dict, Any, Tuple, Optional

import aiohttp

from config import Config

# Cache: nickname_lower -> (expires_at, data)
_faceit_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_faceit_sem = asyncio.Semaphore(Config.FACEIT_MAX_CONCURRENCY)


async def get_player(session: aiohttp.ClientSession, nickname: str) -> Dict[str, Any]:
    """
    Get FACEIT player data by nickname.
    Uses caching to avoid rate limits.
    """
    if not Config.FACEIT_API_KEY:
        raise RuntimeError("FACEIT_API_KEY is not set")

    nick_key = nickname.strip().lower()
    now = time.time()

    # Check cache
    cached = _faceit_cache.get(nick_key)
    if cached and cached[0] > now:
        return cached[1]

    # Make API request
    url = f"{Config.FACEIT_BASE}/players"
    headers = {"Authorization": f"Bearer {Config.FACEIT_API_KEY}"}

    async with _faceit_sem:
        async with session.get(
            url,
            params={"nickname": nickname},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 404:
                raise ValueError(f"FACEIT user not found: {nickname}")
            if resp.status == 401:
                raise RuntimeError("FACEIT unauthorized (check FACEIT_API_KEY)")
            if resp.status == 429:
                raise RuntimeError(
                    "FACEIT rate limited (429). Try later or increase cache TTL / reduce calls."
                )
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"FACEIT error {resp.status}: {text[:200]}")
            data = await resp.json()

    # Update cache
    _faceit_cache[nick_key] = (now + Config.FACEIT_CACHE_TTL_SEC, data)
    return data


def extract_elo_and_level(player_json: Dict[str, Any], game: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract Elo and level from FACEIT player JSON for a specific game."""
    games = player_json.get("games") or {}
    g = games.get(game) or {}
    elo = g.get("faceit_elo")
    lvl = g.get("skill_level")
    
    try:
        elo_int = int(elo) if elo is not None else None
    except (ValueError, TypeError):
        elo_int = None
    
    try:
        lvl_int = int(lvl) if lvl is not None else None
    except (ValueError, TypeError):
        lvl_int = None
    
    return elo_int, lvl_int
