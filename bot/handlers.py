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
        link_faceit(message.chat.id, message.from_user.id, nickname)

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

        async with aiohttp.ClientSession() as session:
            results = []
            errors = []

            async def fetch_one(user_id: int, nick: str):
                try:
                    data = await get_player(session, nick)
                    elo, lvl = extract_elo_and_level(data, Config.FACEIT_GAME)
                    results.append((nick, elo, lvl))
                except Exception as e:
                    logger.warning(f"Error fetching FACEIT data for {nick}: {e}")
                    errors.append((nick, str(e)))

            await asyncio.gather(*(fetch_one(uid, nick) for uid, nick in links))

        # Sort: higher elo first; None last
        def sort_key(item):
            nick, elo, lvl = item
            return (elo is None, -(elo or 0), nick.lower())

        results.sort(key=sort_key)

        lines = [f"🎮 <b>FACEIT Elo</b> (гра: <b>{html.escape(Config.FACEIT_GAME)}</b>)", ""]
        for i, (nick, elo, lvl) in enumerate(results, 1):
            nick_safe = html.escape(nick)
            if elo is None and lvl is None:
                lines.append(f"{i}. {nick_safe}: — (нема даних по {html.escape(Config.FACEIT_GAME)})")
            else:
                elo_txt = "—" if elo is None else str(elo)
                lvl_txt = "—" if lvl is None else str(lvl)
                lines.append(f"{i}. {nick_safe}: <b>{elo_txt}</b> Elo, lvl <b>{lvl_txt}</b>")

        if errors:
            lines.append("")
            lines.append("⚠️ <b>Проблеми:</b>")
            # Show up to 5 errors to avoid spam
            for nick, err in errors[:5]:
                lines.append(f"• {html.escape(nick)}: {html.escape(err[:100])}")
            if len(errors) > 5:
                lines.append(f"• … ще {len(errors) - 5}")

        lines.append("")
        lines.append(f"ℹ️ Кеш: {Config.FACEIT_CACHE_TTL_SEC}с, паралельність: {Config.FACEIT_MAX_CONCURRENCY}.")
        await message.reply("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in /elo command: {e}", exc_info=True)
        await message.reply("❌ Помилка при отриманні Elo. Спробуйте пізніше.")
