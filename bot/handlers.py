"""Bot message handlers."""
import asyncio
import html
import logging

import aiohttp
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from bot.database import (
    add_message,
    top,
    link_faceit,
    unlink_faceit,
    get_faceit_links,
    get_faceit_link,
)
from bot.faceit import get_player, extract_elo_and_level
from bot.utils import estimate_seconds, fmt_duration
from config import Config

logger = logging.getLogger(__name__)

router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message):
    """Handle regular text messages - count statistics."""
    if not message.from_user:
        return

    try:
        u = message.from_user
        full_name = " ".join([p for p in [u.first_name, u.last_name] if p]).strip() or "Unknown"
        text_len = len(message.text or "")

        add_message(
            chat_id=message.chat.id,
            user_id=u.id,
            username=u.username,
            full_name=full_name,
            text_len=text_len
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)


@router.message(Command("stats"))
async def stats(message: Message):
    """Show top-20 statistics for the chat."""
    try:
        rows = top(message.chat.id, 20)
        if not rows:
            await message.reply("Поки нема даних — я рахую з моменту, як мене додали 🙂")
            return

        lines = ["📊 <b>Статистика (топ-20)</b>", ""]
        for i, (who, msg_count, char_count) in enumerate(rows, 1):
            who_safe = html.escape(str(who))
            sec = estimate_seconds(msg_count, char_count)
            lines.append(
                f"{i}. {who_safe}: <b>{msg_count}</b> msg, {char_count} chars, ≈ <b>{fmt_duration(sec)}</b>"
            )

        lines.append("")
        lines.append(
            f"ℹ️ Оцінка: {Config.TYPING_CHARS_PER_MIN} chars/хв + {Config.SECONDS_OVERHEAD_PER_MSG}с/повідомлення."
        )

        await message.reply("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in /stats command: {e}", exc_info=True)
        await message.reply("❌ Помилка при отриманні статистики. Спробуйте пізніше.")


@router.message(Command("linkfaceit"))
async def cmd_linkfaceit(message: Message):
    """Link FACEIT nickname to user."""
    try:
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await message.reply("Використання: <code>/linkfaceit Nickname</code>", parse_mode="HTML")
            return

        nickname = parts[1].strip()
        
        # Check if user already has a linked FACEIT account
        existing_nick = get_faceit_link(message.chat.id, message.from_user.id)
        logger.info(f"linkfaceit: chat_id={message.chat.id}, user_id={message.from_user.id}, existing={existing_nick}, new={nickname}")
        if existing_nick:
            existing_safe = html.escape(existing_nick)
            await message.reply(
                f"❌ У тебе вже додано 1 акаунт FACEIT — <b>{existing_safe}</b>\n"
                f"Ймовірно, це смурф-акаунт, тому я передам його на перевірку для можливого блокування.\n"
                f"Спільнота FACEIT дякує тобі за сприяння чесній грі.",
                parse_mode="HTML"
            )
            return
        
        link_faceit(message.chat.id, message.from_user.id, nickname)

        # Special message for senaToR_cfg
        if nickname.lower() == "senator_cfg":
            await message.reply(
                "ОПА, ЇБАЛА ЖАБА ГАДЮКУ, МЄНТ З МОМЕНТАЛКОЙ",
                parse_mode="HTML"
            )
            return

        await message.reply(
            f"✅ Прив'язав FACEIT нік: <b>{html.escape(nickname)}</b>\n"
            f"Тепер команда <code>/elo</code> покаже твій Elo (гра: <b>{html.escape(Config.FACEIT_GAME)}</b>).",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error in /linkfaceit command: {e}", exc_info=True)
        await message.reply("❌ Помилка при прив'язці ніку. Спробуйте пізніше.")


@router.message(Command("unlinkfaceit"))
async def cmd_unlinkfaceit(message: Message):
    """Unlink FACEIT nickname from user."""
    try:
        ok = unlink_faceit(message.chat.id, message.from_user.id)
        if ok:
            await message.reply("🧹 Відв'язав FACEIT нік.", parse_mode="HTML")
        else:
            await message.reply(
                "У тебе не було прив'язаного FACEIT ніку. /linkfaceit <nickname>",
                parse_mode="HTML"
            )
    except Exception as e:
        logger.error(f"Error in /unlinkfaceit command: {e}", exc_info=True)
        await message.reply("❌ Помилка при відв'язці ніку. Спробуйте пізніше.")


@router.message(Command("elo"))
async def cmd_elo(message: Message):
    """Show FACEIT Elo rankings for linked users."""
    try:
        if not Config.FACEIT_API_KEY:
            await message.reply(
                "❗️ FACEIT інтеграція не налаштована.\n"
                "Додай змінну середовища <code>FACEIT_API_KEY</code> і перезапусти бота.",
                parse_mode="HTML"
            )
            return

        links = get_faceit_links(message.chat.id)
        if not links:
            await message.reply(
                "Ніхто ще не прив'язав FACEIT нік.\n"
                "Зробіть: <code>/linkfaceit Nickname</code>",
                parse_mode="HTML"
            )
            return

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = []

            async def fetch_one(user_id: int, nick: str):
                try:
                    data = await get_player(session, nick)
                    elo, lvl = extract_elo_and_level(data, Config.FACEIT_GAME)
                    results.append((nick, elo, lvl, None))  # (nick, elo, lvl, error)
                except ValueError as e:
                    # User not found - show immediately
                    error_msg = str(e)
                    if "not found" in error_msg.lower():
                        error_msg = f"Користувача не знайдено"
                    logger.warning(f"FACEIT user not found: {nick}")
                    results.append((nick, None, None, error_msg))
                except RuntimeError as e:
                    # API errors
                    error_msg = str(e)
                    if "rate limited" in error_msg.lower():
                        error_msg = "Перевищено ліміт запитів"
                    elif "unauthorized" in error_msg.lower():
                        error_msg = "Помилка авторизації API"
                    else:
                        error_msg = "Помилка FACEIT API"
                    logger.warning(f"Error fetching FACEIT data for {nick}: {e}")
                    results.append((nick, None, None, error_msg))
                except Exception as e:
                    # Other errors - show user-friendly message
                    error_type = type(e).__name__
                    logger.warning(f"Error fetching FACEIT data for {nick}: {e}", exc_info=True)
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        error_msg = "Таймаут запиту"
                    elif "connection" in error_str:
                        error_msg = "Помилка підключення"
                    else:
                        error_msg = "Помилка отримання даних"
                    results.append((nick, None, None, error_msg))

            await asyncio.gather(*(fetch_one(uid, nick) for uid, nick in links), return_exceptions=True)

        # Sort: higher elo first; None last; errors at the end
        def sort_key(item):
            nick, elo, lvl, error = item
            if error:
                return (2, 0, nick.lower())  # Errors go last
            return (elo is None, -(elo or 0), nick.lower())

        results.sort(key=sort_key)

        lines = [f"🎮 <b>FACEIT Elo</b> (гра: <b>{html.escape(Config.FACEIT_GAME)}</b>)", ""]
        for i, (nick, elo, lvl, error) in enumerate(results, 1):
            nick_safe = html.escape(nick)
            if error:
                # Show error immediately - clean up technical details
                error_msg = str(error)
                # Remove technical details like "Task", file paths, etc.
                if "Task" in error_msg or "coro=" in error_msg or "/Users/" in error_msg:
                    # Extract meaningful part or use generic message
                    if "not found" in error_msg.lower():
                        error_msg = "Користувача не знайдено"
                    elif "timeout" in error_msg.lower():
                        error_msg = "Таймаут запиту"
                    else:
                        error_msg = "Помилка отримання даних"
                # Limit length and escape
                error_safe = html.escape(error_msg[:80])
                lines.append(f"{i}. {nick_safe}: ❌ {error_safe}")
            elif elo is None and lvl is None:
                lines.append(f"{i}. {nick_safe}: — (нема даних по {html.escape(Config.FACEIT_GAME)})")
            else:
                elo_txt = "—" if elo is None else str(elo)
                lvl_txt = "—" if lvl is None else str(lvl)
                lines.append(f"{i}. {nick_safe}: <b>{elo_txt}</b> Elo, lvl <b>{lvl_txt}</b>")

        lines.append("")
        lines.append(f"ℹ️ Кеш: {Config.FACEIT_CACHE_TTL_SEC}с, паралельність: {Config.FACEIT_MAX_CONCURRENCY}.")
        await message.reply("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in /elo command: {e}", exc_info=True)
        await message.reply("❌ Помилка при отриманні Elo. Спробуйте пізніше.")
