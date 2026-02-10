"""Daily summary generator."""
import html
import logging
import re
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional

import aiohttp

from bot.database import (
    get_conn,
    get_faceit_links,
    save_elo_history,
    get_previous_elo,
    get_today_messages,
)
from bot.faceit import get_player, extract_elo_and_level
from bot.utils import estimate_seconds, fmt_duration
from bot.topic_analyzer import generate_topic_summary, generate_text_summary, count_mentions
from config import Config

logger = logging.getLogger(__name__)


def get_daily_stats(chat_id: int, target_date: Optional[date] = None) -> List[Tuple[str, int, int]]:
    """Get daily statistics for a chat. Returns list of (username, msg_count, char_count)."""
    if target_date is None:
        target_date = date.today()
    
    # Get stats from user_stats filtered by last_message_date, but count from messages for accuracy
    date_str = target_date.isoformat()
    
    conn = get_conn()
    try:
        cur = conn.cursor()
        # Get users from user_stats who wrote today (filter by last_message_date)
        # But count actual messages from messages table for accuracy
        cur.execute("""
            SELECT
                COALESCE(NULLIF(us.username, ''), us.full_name, 'Unknown') as who,
                COUNT(m.id) as msg_count,
                SUM(LENGTH(m.message_text)) as char_count
            FROM user_stats us
            LEFT JOIN messages m ON us.chat_id = m.chat_id 
                AND us.user_id = m.user_id 
                AND date(m.created_at) = date(?)
            WHERE us.chat_id = ? 
                AND us.last_message_date = ?
            GROUP BY us.user_id, us.username, us.full_name
            HAVING msg_count > 0
            ORDER BY msg_count DESC, char_count DESC
            LIMIT 20
        """, (date_str, chat_id, date_str))
        return cur.fetchall()
    finally:
        conn.close()


async def generate_daily_summary(chat_id: int, bot=None) -> str:
    """Generate daily summary for a chat."""
    logger.info(f"Starting summary generation for chat {chat_id}")
    try:
        lines = [f"📊 <b>Щоденний звіт</b> — {date.today().strftime('%d.%m.%Y')}", ""]
        logger.info(f"Summary header created")
    except Exception as e:
        logger.error(f"Error generating summary header: {e}", exc_info=True)
        lines = [f"📊 <b>Щоденний звіт</b> — {date.today().strftime('%d.%m.%Y')}", ""]
    
    # Get top users
    try:
        stats = get_daily_stats(chat_id)
    except Exception as e:
        logger.warning(f"Error getting daily stats: {e}")
        stats = []
    
    if stats:
        lines.append("🏆 <b>Топ активних користувачів за сьогодні</b>")
        for i, (who, msg_count, char_count) in enumerate(stats[:10], 1):
            who_safe = html.escape(str(who))
            sec = estimate_seconds(msg_count, char_count)
            lines.append(
                f"{i}. {who_safe}: <b>{msg_count}</b> повідомлень, ≈ <b>{fmt_duration(sec)}</b>"
            )
        lines.append("")
    
    # Get FACEIT Elo changes
    if Config.FACEIT_API_KEY:
        links = get_faceit_links(chat_id)
        if links:
            elo_changes = []
            connector = aiohttp.TCPConnector(limit=10)
            async with aiohttp.ClientSession(connector=connector) as session:
                for user_id, nick in links:
                    try:
                        data = await get_player(session, nick)
                        elo, lvl = extract_elo_and_level(data, Config.FACEIT_GAME)
                        if elo is not None:
                            prev_elo = get_previous_elo(nick)
                            if prev_elo is not None:
                                diff = elo - prev_elo
                                if diff != 0:
                                    elo_changes.append((nick, elo, diff))
                            save_elo_history(nick, elo)
                    except Exception as e:
                        logger.warning(f"Error fetching FACEIT data for {nick} in summary: {e}")
            
            if elo_changes:
                lines.append("🎮 <b>Зміни FACEIT Elo:</b>")
                for nick, elo, diff in sorted(elo_changes, key=lambda x: abs(x[2]), reverse=True):
                    nick_safe = html.escape(nick)
                    diff_txt = f"+{diff}" if diff > 0 else str(diff)
                    lines.append(f"• {nick_safe}: <b>{elo}</b> ({diff_txt})")
                lines.append("")
    
    # Generate detailed text summary by topics
    messages = []
    try:
        messages = get_today_messages(chat_id)
        logger.info(f"Retrieved {len(messages)} messages for summary")
        if len(messages) >= 1:  # Summarize if there is at least 1 message
            text_summary = await generate_text_summary(messages)
            logger.info(f"Generated text summary: {text_summary[:100] if text_summary else 'None'}...")
            if text_summary:
                lines.append("")
                lines.append("📝 <b>Детальний зміст обговорення:</b>")
                lines.append("")
                lines.append(text_summary)
            else:
                logger.warning("text_summary is None or empty")
    except Exception as e:
        logger.warning(f"Error getting messages for summary: {e}", exc_info=True)
        messages = []
    
    # Мєнт счьотчік
    try:
        if messages:
            # Count mentions of "мєнт" (ment) in messages
            mention_count = 0
            for _, _, message_text in messages:
                text_lower = message_text.lower()
                # Count word "мєнт" or "мент" (case-insensitive)
                matches = re.findall(r'\bм[єе]нт\b', text_lower)
                mention_count += len(matches)
                if matches:
                    logger.info(f"Found 'мєнт' in message: {message_text[:50]}...")
            
            logger.info(f"Мєнт счьотчік: found {mention_count} mentions")
            if mention_count > 0:
                lines.append("")
                lines.append(f"🔢 <b>Мєнт счьотчік:</b> @LeOnid_CHINANEN був згаданий як \"мєнт\" <b>{mention_count}</b> разів")
            else:
                logger.info("Мєнт счьотчік: no mentions found")
        else:
            logger.info("No messages for мєнт счьотчік")
    except Exception as e:
        logger.warning(f"Error counting mentions: {e}", exc_info=True)
    
    # If no data at all, show message
    if len(lines) == 2:  # Only header and empty line
        lines.append("ℹ️ Поки немає даних за сьогодні.")
    
    lines.append("")
    lines.append("💬 Гарного дня! 🚀")
    result = "\n".join(lines)
    logger.info(f"Summary generated successfully, total lines: {len(lines)}, length: {len(result)}")
    return result


async def send_daily_summary(chat_id: int, bot):
    """Send daily summary to a chat."""
    try:
        summary = await generate_daily_summary(chat_id, bot)
        await bot.send_message(chat_id, summary, parse_mode="HTML")
        logger.info(f"Sent daily summary to chat {chat_id}")
    except Exception as e:
        logger.error(f"Error sending daily summary to chat {chat_id}: {e}", exc_info=True)
