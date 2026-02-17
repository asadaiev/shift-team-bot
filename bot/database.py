"""Database operations using MongoDB."""
import os
import logging
from datetime import date, datetime
from typing import Optional, List, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from config import Config

logger = logging.getLogger(__name__)

# MongoDB client and database
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_client() -> MongoClient:
    """Get MongoDB client (singleton)."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    return _client


def get_db() -> Database:
    """Get MongoDB database (singleton)."""
    global _db
    if _db is None:
        client = get_client()
        # Extract database name from URI or use default
        uri = Config.MONGODB_URI
        db_name = "mybot"  # Default
        
        # Parse URI: mongodb+srv://user:pass@host/dbname?options
        # or: mongodb://user:pass@host:port/dbname?options
        try:
            if "/" in uri:
                # Split by / and get the part after the last /
                parts = uri.split("/")
                if len(parts) >= 4:  # mongodb+srv://user:pass@host/dbname?options
                    db_name = parts[-1].split("?")[0]
                    # Validate db_name
                    if not db_name or db_name == "" or db_name == uri:
                        db_name = "mybot"
                        logger.warning(f"Could not extract database name from URI, using default: {db_name}")
                else:
                    logger.warning(f"URI format unexpected, using default database: {db_name}")
            else:
                logger.warning(f"No database name in URI, using default: {db_name}")
        except Exception as e:
            logger.warning(f"Error parsing database name from URI: {e}, using default: {db_name}")
        
        _db = client[db_name]
        logger.info(f"Using MongoDB database: '{db_name}'")
        
        # Test database access and log collection info
        try:
            collections = _db.list_collection_names()
            logger.info(f"MongoDB database access verified. Collections: {collections}")
            
            # Log document counts for debugging
            for coll_name in ["user_stats", "faceit_links", "messages", "faceit_elo_history"]:
                count = _db[coll_name].count_documents({})
                if count > 0:
                    logger.info(f"Collection '{coll_name}': {count} documents")
        except Exception as e:
            logger.error(f"Error accessing MongoDB database: {e}", exc_info=True)
            raise
    
    return _db


def get_collection(name: str) -> Collection:
    """Get a MongoDB collection."""
    db = get_db()
    return db[name]


def add_message(chat_id: int, user_id: int, username: Optional[str], full_name: str, text_len: int, message_text: Optional[str] = None) -> None:
    """Add or update message statistics for a user."""
    try:
        user_stats = get_collection("user_stats")
        messages = get_collection("messages")
        
        now = datetime.now()
        today_str = date.today().isoformat()
        
        # Update user statistics
        result = user_stats.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {
                "$set": {
                    "username": username,
                    "full_name": full_name,
                    "last_message_date": today_str
                },
                "$inc": {
                    "msg_count": 1,
                    "char_count": text_len
                }
            },
            upsert=True
        )
        
        if result.upserted_id:
            logger.debug(f"Created new user_stats entry for chat_id={chat_id}, user_id={user_id}")
        else:
            logger.debug(f"Updated user_stats for chat_id={chat_id}, user_id={user_id}")
        
        # Store message text for topic analysis (only if provided and not too long)
        if message_text and len(message_text) > 0 and len(message_text) <= 2000:
            msg_result = messages.insert_one({
                "chat_id": chat_id,
                "user_id": user_id,
                "message_text": message_text,
                "created_at": now.isoformat()
            })
            logger.debug(f"Inserted message with id={msg_result.inserted_id}")
    except Exception as e:
        logger.error(f"Error adding message to MongoDB: {e}", exc_info=True)
        raise


def get_today_messages(chat_id: int) -> List[Tuple[str, str, str]]:
    """Get all messages from today for a chat. Returns list of (user_id, username_or_fullname, message_text)."""
    return get_messages_by_date(chat_id, date.today())


def get_messages_by_date(chat_id: int, target_date: date) -> List[Tuple[str, str, str]]:
    """Get all messages for a specific date. Returns list of (user_id, username_or_fullname, message_text)."""
    from datetime import time as dt_time
    
    messages = get_collection("messages")
    user_stats = get_collection("user_stats")
    
    date_str = target_date.isoformat()
    start_of_day = datetime.combine(target_date, dt_time.min)
    end_of_day = datetime.combine(target_date, dt_time.max)
    
    # Get messages for the date
    message_docs = messages.find({
        "chat_id": chat_id,
        "created_at": {
            "$gte": start_of_day.isoformat(),
            "$lte": end_of_day.isoformat()
        }
    }).sort("created_at", 1)
    
    # Build result with usernames
    result = []
    for msg in message_docs:
        user_id_str = str(msg["user_id"])
        
        # Get username from user_stats
        user_stat = user_stats.find_one({
            "chat_id": chat_id,
            "user_id": msg["user_id"]
        })
        
        if user_stat:
            username = user_stat.get("username") or user_stat.get("full_name") or "Unknown"
        else:
            username = "Unknown"
        
        result.append((user_id_str, username, msg["message_text"]))
    
    return result


def get_active_chats_today() -> List[int]:
    """Get all chat IDs that have activity today."""
    user_stats = get_collection("user_stats")
    today_str = date.today().isoformat()
    
    # Find distinct chat_ids where last_message_date = today
    chat_ids = user_stats.distinct("chat_id", {"last_message_date": today_str})
    return list(chat_ids)


def top(chat_id: int, limit: int = 20) -> List[Tuple[str, int, int]]:
    """Get top users by message count for a chat."""
    user_stats = get_collection("user_stats")
    
    cursor = user_stats.find(
        {"chat_id": chat_id}
    ).sort([
        ("msg_count", -1),
        ("char_count", -1)
    ]).limit(limit)
    
    result = []
    for doc in cursor:
        who = doc.get("username") or doc.get("full_name") or "Unknown"
        msg_count = doc.get("msg_count", 0)
        char_count = doc.get("char_count", 0)
        result.append((who, msg_count, char_count))
    
    return result


def link_faceit(chat_id: int, user_id: int, nickname: str) -> None:
    """Link a FACEIT nickname to a user in a chat."""
    faceit_links = get_collection("faceit_links")
    
    faceit_links.update_one(
        {"chat_id": chat_id, "user_id": user_id},
        {"$set": {"nickname": nickname}},
        upsert=True
    )


def unlink_faceit(chat_id: int, user_id: int) -> bool:
    """Unlink FACEIT nickname for a user. Returns True if unlinked."""
    faceit_links = get_collection("faceit_links")
    
    result = faceit_links.delete_one({"chat_id": chat_id, "user_id": user_id})
    return result.deleted_count > 0


def unlink_faceit_by_nickname(nickname: str) -> int:
    """Unlink FACEIT nickname by nickname (admin function). Returns number of deleted records."""
    faceit_links = get_collection("faceit_links")
    
    result = faceit_links.delete_many({"nickname": {"$regex": f"^{nickname}$", "$options": "i"}})
    return result.deleted_count


def get_faceit_link(chat_id: int, user_id: int) -> Optional[str]:
    """Get FACEIT nickname for a specific user in a chat. Returns None if not linked."""
    faceit_links = get_collection("faceit_links")
    
    doc = faceit_links.find_one({"chat_id": chat_id, "user_id": user_id})
    return doc["nickname"] if doc else None


def get_faceit_links(chat_id: int) -> List[Tuple[int, str]]:
    """Get all FACEIT links for a chat. Returns list of (user_id, nickname)."""
    faceit_links = get_collection("faceit_links")
    
    cursor = faceit_links.find({"chat_id": chat_id}).sort("nickname", 1)
    
    result = []
    for doc in cursor:
        result.append((doc["user_id"], doc["nickname"]))
    
    return result


def get_user_id_by_username(username: str) -> Optional[int]:
    """Get user_id by username from user_stats table. Returns None if not found."""
    user_stats = get_collection("user_stats")
    
    doc = user_stats.find_one({"username": username})
    return doc["user_id"] if doc else None


def save_elo_history(nickname: str, elo: Optional[int]) -> None:
    """Save Elo value for today's date."""
    if elo is None:
        return
    
    elo_history = get_collection("faceit_elo_history")
    today_str = date.today().isoformat()
    
    elo_history.update_one(
        {"nickname": nickname.lower(), "date": today_str},
        {"$set": {"elo": elo}},
        upsert=True
    )


def get_previous_elo(nickname: str) -> Optional[int]:
    """Get last saved Elo value (from any date, excluding today if it exists). Returns None if no history."""
    elo_history = get_collection("faceit_elo_history")
    today_str = date.today().isoformat()
    
    # Get the most recent Elo that is NOT from today
    doc = elo_history.find_one(
        {"nickname": nickname.lower(), "date": {"$lt": today_str}},
        sort=[("date", -1)]
    )
    
    if doc:
        return doc.get("elo")
    
    # If no previous dates, get the last saved value regardless of date
    doc = elo_history.find_one(
        {"nickname": nickname.lower()},
        sort=[("date", -1)]
    )
    
    return doc.get("elo") if doc else None
