"""Database operations."""
import sqlite3
from datetime import date, datetime
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
            last_message_date TEXT,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    # Add last_message_date column if it doesn't exist (migration)
    try:
        conn.execute("ALTER TABLE user_stats ADD COLUMN last_message_date TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faceit_links (
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            nickname TEXT NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faceit_elo_history (
            nickname TEXT NOT NULL,
            date TEXT NOT NULL,
            elo INTEGER NOT NULL,
            PRIMARY KEY (nickname, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            message_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_date ON messages(chat_id, created_at)
    """)
    return conn


def add_message(chat_id: int, user_id: int, username: Optional[str], full_name: str, text_len: int, message_text: Optional[str] = None) -> None:
    """Add or update message statistics for a user."""
    conn = get_conn()
    try:
        with conn:
            # Update statistics
            now = datetime.now().isoformat()
            today_str = date.today().isoformat()
            conn.execute("""
                INSERT INTO user_stats (chat_id, user_id, username, full_name, msg_count, char_count, last_message_date)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    username=excluded.username,
                    full_name=excluded.full_name,
                    msg_count=user_stats.msg_count + 1,
                    char_count=user_stats.char_count + excluded.char_count,
                    last_message_date=excluded.last_message_date
            """, (chat_id, user_id, username, full_name, text_len, today_str))
            
            # Store message text for topic analysis (only if provided and not too long)
            if message_text and len(message_text) > 0 and len(message_text) <= 2000:
                conn.execute("""
                    INSERT INTO messages (chat_id, user_id, message_text, created_at)
                    VALUES (?, ?, ?, ?)
                """, (chat_id, user_id, message_text, now))
    finally:
        conn.close()


def get_today_messages(chat_id: int) -> List[Tuple[str, str, str]]:
    """Get all messages from today for a chat. Returns list of (user_id, username_or_fullname, message_text)."""
    conn = get_conn()
    try:
        today = date.today().isoformat()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                m.user_id,
                COALESCE(NULLIF(us.username, ''), us.full_name, 'Unknown') as who,
                m.message_text
            FROM messages m
            LEFT JOIN user_stats us ON m.chat_id = us.chat_id AND m.user_id = us.user_id
            WHERE m.chat_id = ? AND date(m.created_at) = date(?)
            ORDER BY m.created_at
        """, (chat_id, today))
        return cur.fetchall()
    finally:
        conn.close()


def get_active_chats_today() -> List[int]:
    """Get all chat IDs that have activity today (from user_stats where last_message_date = today)."""
    conn = get_conn()
    try:
        today = date.today().isoformat()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT chat_id
            FROM user_stats
            WHERE last_message_date = ?
        """, (today,))
        return [row[0] for row in cur.fetchall()]
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


def unlink_faceit_by_nickname(nickname: str) -> int:
    """Unlink FACEIT nickname by nickname (admin function). Returns number of deleted records."""
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute("""
                DELETE FROM faceit_links
                WHERE LOWER(nickname) = LOWER(?)
            """, (nickname,))
            return cur.rowcount
    finally:
        conn.close()


def get_faceit_link(chat_id: int, user_id: int) -> Optional[str]:
    """Get FACEIT nickname for a specific user in a chat. Returns None if not linked."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT nickname
            FROM faceit_links
            WHERE chat_id = ? AND user_id = ?
        """, (chat_id, user_id))
        row = cur.fetchone()
        return row[0] if row else None
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


def save_elo_history(nickname: str, elo: Optional[int]) -> None:
    """Save Elo value for today's date."""
    if elo is None:
        return
    
    conn = get_conn()
    try:
        today = date.today().isoformat()
        with conn:
            conn.execute("""
                INSERT INTO faceit_elo_history (nickname, date, elo)
                VALUES (?, ?, ?)
                ON CONFLICT(nickname, date) DO UPDATE SET
                    elo=excluded.elo
            """, (nickname.lower(), today, elo))
    finally:
        conn.close()


def get_previous_elo(nickname: str) -> Optional[int]:
    """Get last saved Elo value (from any date, excluding today if it exists). Returns None if no history."""
    conn = get_conn()
    try:
        today = date.today().isoformat()
        
        cur = conn.cursor()
        # Get the most recent Elo that is NOT from today (to compare with last game)
        cur.execute("""
            SELECT elo FROM faceit_elo_history
            WHERE nickname = ? AND date < ?
            ORDER BY date DESC
            LIMIT 1
        """, (nickname.lower(), today))
        row = cur.fetchone()
        if row:
            return row[0]
        
        # If no previous dates, check if today's value exists (but we'll use it only if it's different from current)
        # Actually, let's get the last saved value regardless of date, but exclude the current one
        # For simplicity, we'll get the last saved value before saving the new one
        cur.execute("""
            SELECT elo FROM faceit_elo_history
            WHERE nickname = ?
            ORDER BY date DESC, rowid DESC
            LIMIT 1
        """, (nickname.lower(),))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()
