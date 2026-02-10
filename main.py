"""Main entry point for the bot."""
import asyncio
import logging
import sys
from datetime import datetime, time, timedelta

from aiogram import Bot, Dispatcher

from bot.handlers import router
from bot.database import get_active_chats_today
from bot.summary import send_daily_summary
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


async def daily_summary_scheduler(bot: Bot):
    """Schedule daily summary to be sent at the end of the day."""
    summary_time = time(23, 0)  # 23:00 (11 PM)
    
    while True:
        now = datetime.now().time()
        target = datetime.combine(datetime.now().date(), summary_time)
        
        # Calculate seconds until summary time
        if now < summary_time:
            # Today at summary_time
            wait_until = target
        else:
            # Tomorrow at summary_time
            wait_until = target + timedelta(days=1)
        
        wait_seconds = (wait_until - datetime.now()).total_seconds()
        logger.info(f"Next daily summary scheduled in {wait_seconds/3600:.1f} hours")
        
        await asyncio.sleep(wait_seconds)
        
        # Send summary to all chats with activity today
        active_chats = get_active_chats_today()
        if active_chats:
            logger.info(f"Sending daily summary to {len(active_chats)} chats")
            for chat_id in active_chats:
                try:
                    await send_daily_summary(chat_id, bot)
                except Exception as e:
                    logger.error(f"Error sending summary to chat {chat_id}: {e}")
        else:
            logger.info("No active chats today, skipping summary")
        
        # Wait a bit before next check
        await asyncio.sleep(60)


async def main():
    """Start the bot."""
    logger.info("Starting bot...")
    
    bot = Bot(Config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    # Start daily summary scheduler
    scheduler_task = asyncio.create_task(daily_summary_scheduler(bot))
    
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        scheduler_task.cancel()
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        scheduler_task.cancel()
        raise
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
