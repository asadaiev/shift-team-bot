"""Database operations."""
import sqlite3
from typing import Optional, List, Tuple

from config import Config


def get_conn() -> sqlite3.Connection:
    """Get database connection with timeout to avoid locks."""
    conn = sqlite3.connect(Config.DB_PATH, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT,
            full_name TEXT,
            msg_count INTEGER NOT NULL DEFAULT 0,
            char_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faceit_links (
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            nickname TEXT NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    return conn


def add_message(chat_id: int, user_id: int, username: Optional[str], full_name: str, text_len: int) -> None:
    """Add or update message statistics for a user."""
    conn = get_conn()
    try:
        with conn:
            conn.execute("""
                INSERT INTO user_stats (chat_id, user_id, username, full_name, msg_count, char_count)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    username=excluded.username,
                    full_name=excluded.full_name,
                    msg_count=user_stats.msg_count + 1,
                    char_count=user_stats.char_count + excluded.char_count
            """, (chat_id, user_id, username, full_name, text_len))
    finally:
        conn.close()


def top(chat_id: int, limit: int = 20) -> List[Tuple[str, int, int]]:
    """Get top users by message count for a chat."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COALESCE(NULLIF(username, ''), full_name) as who,
                msg_count,
                char_count
            FROM user_stats
            WHERE chat_id = ?
            ORDER BY msg_count DESC, char_count DESC
            LIMIT ?
        """, (chat_id, limit))
        return cur.fetchall()
    finally:
        conn.close()


def link_faceit(chat_id: int, user_id: int, nickname: str) -> None:
    """Link a FACEIT nickname to a user in a chat."""
    conn = get_conn()
    try:
        with conn:
            conn.execute("""
                INSERT INTO faceit_links (chat_id, user_id, nickname)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    nickname=excluded.nickname
            """, (chat_id, user_id, nickname))
    finally:
        conn.close()


def unlink_faceit(chat_id: int, user_id: int) -> bool:
    """Unlink FACEIT nickname for a user. Returns True if unlinked."""
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute("""
                DELETE FROM faceit_links
                WHERE chat_id = ? AND user_id = ?
            """, (chat_id, user_id))
            return cur.rowcount > 0
    finally:
        conn.close()


def get_faceit_links(chat_id: int) -> List[Tuple[int, str]]:
    """Get all FACEIT links for a chat. Returns list of (user_id, nickname)."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT user_id, nickname
            FROM faceit_links
            WHERE chat_id = ?
            ORDER BY nickname COLLATE NOCASE
        """, (chat_id,))
        return cur.fetchall()
    finally:
        conn.close()
