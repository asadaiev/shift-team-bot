"""Main entry point for the bot."""
import asyncio
import logging
import os
import sys
import threading
from datetime import datetime, time, timedelta

from aiogram import Bot, Dispatcher
from flask import Flask

from bot.handlers import router
from bot.database import get_active_chats_today, get_user_id_by_username
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

# Flask app for health check
app = Flask(__name__)

@app.route("/")
def home():
    """Health check endpoint for Render."""
    return "Bot is running"

def run_flask():
    """Run Flask server in a separate thread."""
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


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
        
        # Send summary only to admin
        # Try to get admin user_id from config or database
        admin_user_id = Config.ADMIN_USER_ID
        if not admin_user_id:
            # Try to get from database by username
            admin_user_id = get_user_id_by_username(Config.ADMIN_USERNAME)
            if admin_user_id:
                logger.info(f"Found admin user_id {admin_user_id} from database for username {Config.ADMIN_USERNAME}")
        
        if admin_user_id:
            logger.info(f"Sending daily summary to admin (user_id: {admin_user_id}, username: {Config.ADMIN_USERNAME})")
            try:
                # Get all active chats and generate summary for each
                active_chats = get_active_chats_today()
                if active_chats:
                    for chat_id in active_chats:
                        try:
                            # Send summary to admin for this chat
                            await send_daily_summary(chat_id, bot, send_to_admin=True, admin_user_id=admin_user_id)
                        except Exception as e:
                            logger.error(f"Error sending summary for chat {chat_id} to admin: {e}")
                else:
                    logger.info("No active chats today, skipping summary")
            except Exception as e:
                logger.error(f"Error sending summary to admin: {e}")
        else:
            logger.warning(f"ADMIN_USER_ID not set and admin username '{Config.ADMIN_USERNAME}' not found in database, skipping summary")
        
        # Wait a bit before next check
        await asyncio.sleep(60)


async def main():
    """Start the bot."""
    logger.info("Starting bot...")
    
    # Test MongoDB connection before starting
    try:
        from bot.database import get_db, get_collection
        db = get_db()
        # Try to access a collection to verify connection
        test_collection = get_collection("user_stats")
        test_collection.find_one()  # Simple query to test connection
        logger.info("MongoDB connection verified successfully")
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}", exc_info=True)
        logger.error("Bot will continue, but database operations may fail")
    
    # Start Flask server in a separate thread for health check
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask health check server started")
    
    bot = Bot(Config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    # Start daily summary scheduler
    scheduler_task = asyncio.create_task(daily_summary_scheduler(bot))
    
    try:
        # Try to close any existing webhook first (if any)
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            logger.info("Deleted any existing webhook")
        except Exception as e:
            logger.warning(f"Could not delete webhook (might not exist): {e}")
        
        await dp.start_polling(bot, drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        scheduler_task.cancel()
    except Exception as e:
        error_str = str(e).lower()
        if "conflict" in error_str or "getupdates" in error_str:
            logger.error(
                "⚠️ TelegramConflictError: Another bot instance is running!\n"
                "Please ensure:\n"
                "1. No local bot instance is running (stop with Ctrl+C)\n"
                "2. Only one instance is running on Render\n"
                "3. No webhook is set for this bot token"
            )
            # Don't raise, just log and exit gracefully
            scheduler_task.cancel()
            return
        logger.error(f"Bot error: {e}", exc_info=True)
        scheduler_task.cancel()
        raise
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
