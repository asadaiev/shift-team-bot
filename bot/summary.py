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
    get_messages_by_date,
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


async def generate_daily_summary(chat_id: int, bot=None, target_date: Optional[date] = None) -> str:
    """Generate daily summary for a chat. If target_date is None, uses today."""
    if target_date is None:
        target_date = date.today()
    
    logger.info(f"Starting summary generation for chat {chat_id}, date: {target_date}")
    try:
        lines = [f"📊 <b>Щоденний звіт</b> — {target_date.strftime('%d.%m.%Y')}", ""]
        logger.info(f"Summary header created")
    except Exception as e:
        logger.error(f"Error generating summary header: {e}", exc_info=True)
        lines = [f"📊 <b>Щоденний звіт</b> — {target_date.strftime('%d.%m.%Y')}", ""]
    
    # Get top users
    try:
        stats = get_daily_stats(chat_id, target_date=target_date)
    except Exception as e:
        logger.warning(f"Error getting daily stats: {e}")
        stats = []
    
    if stats:
        date_label = "сьогодні" if target_date == date.today() else target_date.strftime('%d.%m.%Y')
        lines.append(f"🏆 <b>Топ активних користувачів за {date_label}</b>")
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
        if target_date == date.today():
            messages = get_today_messages(chat_id)
        else:
            messages = get_messages_by_date(chat_id, target_date)
        logger.info(f"Retrieved {len(messages)} messages for summary (date: {target_date})")
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
            # Count mentions of "мєнт", "мусор" and similar words in messages
            mention_count = 0
            # Pattern to match: мєнт/мент, мусор, and words with similar roots
            # м[єе]нт - мєнт/мент
            # мусор - мусор
            # м[єе]нт[а-я]* - words starting with мєнт/мент (like мєнтовський, ментівський)
            # мусор[а-я]* - words starting with мусор (like мусорський)
            patterns = [
                r'\bм[єе]нт\b',           # мєнт/мент (exact word)
                r'\bмусор\b',             # мусор (exact word)
                r'\bм[єе]нт[а-яіїє]*\b', # words starting with мєнт/мент
                r'\bмусор[а-яіїє]*\b',   # words starting with мусор
            ]
            
            for _, _, message_text in messages:
                text_lower = message_text.lower()
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    mention_count += len(matches)
                    if matches:
                        logger.info(f"Found mention in message: {message_text[:50]}... (pattern: {pattern})")
            
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


def split_message(text: str, max_length: int = 4096, max_parts: int = 2) -> List[str]:
    """Split a long message into maximum 2 parts that fit within Telegram's limit.
    Tries to split at topic boundaries to avoid cutting topics in half."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split by topics first (lines starting with number and dot, e.g., "1. Topic")
    topic_pattern = re.compile(r'^\d+\.\s+<b>', re.MULTILINE)
    topic_matches = list(topic_pattern.finditer(text))
    
    if len(topic_matches) > 1:
        # Find the middle topic to split approximately in half
        total_length = len(text)
        target_split = total_length // 2
        
        # Find the topic closest to the middle
        best_split_idx = 1
        best_split_pos = topic_matches[1].start()
        min_distance = abs(best_split_pos - target_split)
        
        for i in range(2, len(topic_matches)):
            pos = topic_matches[i].start()
            distance = abs(pos - target_split)
            if distance < min_distance:
                min_distance = distance
                best_split_idx = i
                best_split_pos = pos
        
        # Split at the best position
        part1 = text[:best_split_pos].strip()
        part2 = text[best_split_pos:].strip()
        
        # Check if both parts fit within limit
        if len(part1) <= max_length and len(part2) <= max_length:
            return [part1, part2]
        
        # If one part is still too long, try to split it more carefully
        # But limit to max_parts total
        parts = []
        if len(part1) > max_length:
            # Split part1 by lines
            lines1 = part1.split('\n')
            temp_part = []
            temp_length = 0
            
            for line in lines1:
                line_length = len(line) + 1
                if temp_length + line_length > max_length and temp_part:
                    parts.append('\n'.join(temp_part))
                    if len(parts) >= max_parts - 1:  # Reserve space for part2
                        # Combine remaining lines with part2
                        remaining = '\n'.join(temp_part + lines1[lines1.index(line):])
                        part2 = remaining + '\n' + part2 if remaining else part2
                        break
                    temp_part = [line]
                    temp_length = line_length
                else:
                    temp_part.append(line)
                    temp_length += line_length
            
            if temp_part and len(parts) < max_parts - 1:
                parts.append('\n'.join(temp_part))
        else:
            parts.append(part1)
        
        # Add part2 if we haven't exceeded max_parts
        if len(parts) < max_parts:
            if len(part2) > max_length:
                # Truncate part2 if needed (shouldn't happen often)
                part2 = part2[:max_length - 50] + "\n\n<i>... (повідомлення обрізано через обмеження Telegram)</i>"
            parts.append(part2)
        
        return parts[:max_parts] if parts else [text]
    
    # Fallback: split approximately in half by lines
    lines = text.split('\n')
    total_length = len(text)
    target_split = total_length // 2
    
    # Find the line closest to the middle
    current_length = 0
    split_idx = len(lines) // 2
    
    for i, line in enumerate(lines):
        current_length += len(line) + 1
        if current_length >= target_split:
            split_idx = i
            break
    
    part1 = '\n'.join(lines[:split_idx]).strip()
    part2 = '\n'.join(lines[split_idx:]).strip()
    
    # Check if both parts fit
    if len(part1) <= max_length and len(part2) <= max_length:
        return [part1, part2]
    
    # If still too long, truncate more aggressively
    if len(part1) > max_length:
        part1 = part1[:max_length - 50] + "\n\n<i>... (частина 1)</i>"
    if len(part2) > max_length:
        part2 = part2[:max_length - 50] + "\n\n<i>... (частина 2)</i>"
    
    return [part1, part2]


async def send_daily_summary(chat_id: int, bot, send_to_admin: bool = False, admin_user_id: Optional[int] = None):
    """Send daily summary to a chat or to admin in private message."""
    try:
        summary = await generate_daily_summary(chat_id, bot)
        
        # Split message if it's too long
        parts = split_message(summary, max_length=4096)
        
        # Determine where to send
        if send_to_admin and admin_user_id:
            # Send to admin in private message
            target_id = admin_user_id
            chat_info = f"admin (user_id: {admin_user_id}) for chat {chat_id}"
        else:
            # Send to chat
            target_id = chat_id
            chat_info = f"chat {chat_id}"
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part - add chat info if sending to admin
                if send_to_admin and admin_user_id:
                    header = f"📊 <b>Summary для чату {chat_id}</b>\n\n"
                    await bot.send_message(target_id, header + part, parse_mode="HTML")
                else:
                    await bot.send_message(target_id, part, parse_mode="HTML")
            else:
                # Subsequent parts - add continuation marker
                await bot.send_message(target_id, f"<i>(продовження)</i>\n\n{part}", parse_mode="HTML")
        
        logger.info(f"Sent daily summary to {chat_info} ({len(parts)} parts)")
    except Exception as e:
        logger.error(f"Error sending daily summary to {chat_info if 'chat_info' in locals() else f'chat {chat_id}'}: {e}", exc_info=True)
