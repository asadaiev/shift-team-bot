"""Topic analysis and summarization for messages."""
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional
import logging

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

# Common stop words (Ukrainian and English)
STOP_WORDS = {
    'і', 'та', 'або', 'але', 'що', 'як', 'для', 'від', 'до', 'на', 'з', 'по', 'про', 'за', 'при',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'це', 'той', 'такий', 'який', 'коли', 'де', 'чому', 'якщо', 'хоча', 'тому', 'тому що',
    'так', 'ні', 'не', 'не', 'було', 'буде', 'є', 'був', 'була', 'були',
    'я', 'ти', 'він', 'вона', 'воно', 'ми', 'ви', 'вони',
    'щоб', 'що', 'який', 'яка', 'яке', 'які',
    'це', 'цей', 'ця', 'це', 'ці',
    'той', 'та', 'те', 'ті',
    'м', 'т', 'й', 'ж', 'б', 'ж', 'то', 'а', 'ну', 'о', 'у', 'е', 'и', 'о', 'а'
}

# Minimum word length to consider
MIN_WORD_LENGTH = 3

# Minimum frequency for a topic to be considered (lowered to include more topics)
MIN_TOPIC_FREQUENCY = 1


def extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    # Convert to lowercase and remove special characters
    text = text.lower()
    # Keep only letters, numbers, and Ukrainian characters
    text = re.sub(r'[^\w\s\u0400-\u04FF]', ' ', text)
    words = text.split()
    
    # Filter words
    filtered = []
    for word in words:
        # Remove very short words and stop words
        if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS:
            filtered.append(word)
    
    return filtered


def analyze_topics(messages: List[Tuple[str, str, str]]) -> List[Tuple[str, int]]:
    """Analyze messages and extract main topics. Messages format: (user_id, username, message_text)."""
    if not messages:
        return []
    
    # Extract all words from messages
    all_words = []
    for user_id, username, message_text in messages:
        words = extract_words(message_text)
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get most common words (topics) - increased to 20 for more detail
    topics = word_counts.most_common(20)
    
    # Filter by minimum frequency
    filtered_topics = [(word, count) for word, count in topics if count >= MIN_TOPIC_FREQUENCY]
    
    return filtered_topics


def group_messages_by_topic(messages: List[Tuple[str, str, str]], topics: List[Tuple[str, int]]) -> Dict[str, List[Tuple[str, str]]]:
    """Group messages by detected topics. Returns Dict[topic, (first_mention, List[(username, full_message)])]."""
    topic_groups = {topic: [] for topic, _ in topics}
    topic_first_mention = {}  # Track who first mentioned each topic
    
    for user_id, username, message_text in messages:
        words = set(extract_words(message_text))
        # Find which topics are mentioned in this message
        for topic, _ in topics:
            if topic in words:
                # Track first mention
                if topic not in topic_first_mention:
                    topic_first_mention[topic] = username
                # Store full message (not just snippet) for detailed summary
                topic_groups[topic].append((username, message_text))
                break  # Only assign to first matching topic
    
    # Add first mention info to topic_groups
    for topic in topic_groups:
        if topic in topic_first_mention:
            topic_groups[topic] = (topic_first_mention[topic], topic_groups[topic])
    
    return topic_groups


def count_mentions(messages: List[Tuple[str, str, str]], username: str) -> int:
    """Count how many times a username was mentioned in messages."""
    if not messages:
        return 0
    
    count = 0
    username_lower = username.lower()
    for _, _, message_text in messages:
        # Count mentions (case-insensitive)
        text_lower = message_text.lower()
        # Count as word boundary to avoid partial matches
        count += len(re.findall(r'\b' + re.escape(username_lower) + r'\b', text_lower))
        # Also count @mentions
        count += text_lower.count('@' + username_lower)
    
    return count


def generate_topic_summary(messages: List[Tuple[str, str, str]]) -> str:
    """Generate a summary of topics discussed."""
    if not messages:
        return ""
    
    topics = analyze_topics(messages)
    if not topics:
        return ""
    
    lines = ["💬 <b>Основні теми обговорення:</b>", ""]
    
    for i, (topic, count) in enumerate(topics[:5], 1):  # Top 5 topics
        lines.append(f"{i}. <b>{topic}</b> — згадувалось {count} разів")
    
    return "\n".join(lines)


async def generate_text_summary(messages: List[Tuple[str, str, str]], language: str = "ukrainian") -> Optional[str]:
    """Generate detailed text summary by topics using sumy or OpenAI."""
    if not messages:
        logger.warning("generate_text_summary: no messages provided")
        return None
    
    # Combine all messages into one text
    all_text = "\n".join([msg for _, _, msg in messages])
    logger.info(f"generate_text_summary: combined text length: {len(all_text)}")
    
    if len(all_text) < 50:  # Too short to summarize
        logger.warning(f"generate_text_summary: text too short ({len(all_text)} chars), need at least 50")
        return None
    
    # Try OpenAI first if available and enabled (better quality)
    if Config.USE_OPENAI_SUMMARY and OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
        try:
            logger.info("generate_text_summary: trying OpenAI")
            result = await generate_openai_detailed_summary(all_text)
            if result:
                logger.info(f"generate_text_summary: OpenAI success, length: {len(result)}")
                return result
            else:
                logger.warning("generate_text_summary: OpenAI returned empty result")
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}, falling back to sumy", exc_info=True)
    
    # Fallback to sumy (less detailed but works offline)
    if SUMY_AVAILABLE:
        try:
            logger.info("generate_text_summary: trying sumy")
            result = generate_sumy_detailed_summary(all_text, language)
            if result:
                logger.info(f"generate_text_summary: sumy success, length: {len(result)}")
                return result
            else:
                logger.warning("generate_text_summary: sumy returned empty result")
        except Exception as e:
            logger.warning(f"Sumy summarization failed: {e}", exc_info=True)
    else:
        logger.warning("generate_text_summary: SUMY_AVAILABLE is False, using simple fallback")
        # Simple fallback: generate topic-based summary without sumy
        return generate_simple_fallback_summary(messages)
    
    logger.warning("generate_text_summary: all methods failed, using simple fallback")
    return generate_simple_fallback_summary(messages)


def generate_sumy_summary(text: str, language: str = "ukrainian") -> str:
    """Generate summary using sumy library."""
    if not SUMY_AVAILABLE:
        return ""
    
    try:
        # Parse text
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        
        # Use TextRank summarizer (works better for Ukrainian)
        summarizer = TextRankSummarizer(Stemmer(language))
        summarizer.stop_words = get_stop_words(language)
        
        # Generate summary (3-5 sentences)
        sentence_count = min(5, max(2, len(text.split('.')) // 10))
        summary_sentences = summarizer(parser.document, sentence_count)
        
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating sumy summary: {e}")
        return ""


async def generate_openai_detailed_summary(text: str) -> str:
    """Generate detailed summary by topics using OpenAI API."""
    if not OPENAI_AVAILABLE or not Config.OPENAI_API_KEY:
        return ""
    
    try:
        client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Truncate if too long (OpenAI has token limits)
        if len(text) > 12000:
            text = text[:12000] + "..."
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "Ти допомагаєш створювати дуже детальні summary обговорень українською мовою. Створюй структурований звіт по темах, де кожна тема описується окремим пунктом з максимальними деталями - що саме обговорювалось, які думки висловлювались, які питання піднімались, які висновки робились."
                },
                {
                    "role": "user", 
                    "content": f"Проаналізуй наступне обговорення та створи ДУЖЕ ДЕТАЛЬНИЙ summary українською мовою. Організуй його по темах, де кожна тема - це окремий пункт з максимально детальним описом (4-6 речень на тему). Включи:\n- Що саме обговорювалось\n- Які думки та аргументи висловлювались\n- Які питання піднімались\n- Які висновки або рішення були\n- Хто що говорив (якщо важливо)\n\nМінімум 5-8 пунктів, кожен пункт має бути ДУЖЕ детальним. Формат:\n\n🎯 Тема 1: дуже детальний опис (4-6 речень)\n🎯 Тема 2: дуже детальний опис (4-6 речень)\n\nОбговорення:\n\n{text}"
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating OpenAI summary: {e}")
        return ""


def generate_sumy_detailed_summary(text: str, language: str = "ukrainian") -> str:
    """Generate detailed summary using sumy library (grouped by topics)."""
    if not SUMY_AVAILABLE:
        return ""
    
    try:
        # Parse text
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        
        # Use TextRank summarizer
        summarizer = TextRankSummarizer(Stemmer(language))
        summarizer.stop_words = get_stop_words(language)
        
        # Generate more sentences for detailed summary (increase from 8 to 12-15)
        sentence_count = min(15, max(8, len(text.split('.')) // 5))
        summary_sentences = summarizer(parser.document, sentence_count)
        
        # Group sentences by topics (simple approach: by keywords)
        topics = analyze_topics([("", text)])
        if topics:
            # Create detailed topic-based summary
            lines = []
            for topic, _ in topics[:8]:  # Top 8 topics for more detail
                # Find sentences mentioning this topic
                topic_sentences = [
                    str(s) for s in summary_sentences 
                    if topic.lower() in str(s).lower()
                ]
                if topic_sentences:
                    # Show more sentences per topic (3-4 instead of 2)
                    sentences_text = ' '.join(topic_sentences[:4])
                    lines.append(f"🎯 <b>{topic.capitalize()}</b>: {sentences_text}")
            
            if lines:
                return "\n".join(lines)
        
        # Fallback: detailed summary with all sentences
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating sumy summary: {e}")
        return ""


def generate_simple_fallback_summary(messages: List[Tuple[str, str, str]]) -> str:
    """Generate a detailed topic-based summary without external libraries."""
    if not messages:
        return ""
    
    # Analyze topics
    topics = analyze_topics(messages)
    if not topics:
        # If no topics found, provide detailed summary of all messages
        lines = []
        lines.append("📋 <b>Детальний зміст обговорення:</b>")
        lines.append("")
        for i, (_, username, msg) in enumerate(messages[:10], 1):
            # Show longer snippets (up to 300 chars)
            snippet = msg[:300] + ('...' if len(msg) > 300 else '')
            lines.append(f"{i}. <b>{username}</b>: {snippet}")
        return "\n".join(lines) if lines else ""
    
    # Group messages by topics
    topic_groups = group_messages_by_topic(messages, topics)
    
    # Generate detailed summary by topics
    lines = []
    for topic, topic_data in list(topic_groups.items())[:8]:  # Top 8 topics for more detail
        if isinstance(topic_data, tuple) and len(topic_data) == 2:
            first_mention, topic_messages = topic_data
            if topic_messages:
                # Count unique participants
                participants = set(username for username, _ in topic_messages)
                participant_count = len(participants)
                participant_list = ', '.join(sorted(participants)[:5])
                if participant_count > 5:
                    participant_list += f" та ще {participant_count - 5}"
                
                # Get more messages (up to 5-6) with longer snippets
                lines.append(f"")
                lines.append(f"🎯 <b>{topic.capitalize()}</b> (підняв: <b>{first_mention}</b>, учасників: {participant_count})")
                
                # Show detailed messages (up to 5-6 messages per topic)
                for username, msg_text in topic_messages[:6]:
                    # Show longer snippets (up to 250 chars)
                    snippet = msg_text[:250] + ('...' if len(msg_text) > 250 else '')
                    lines.append(f"   • <b>{username}</b>: {snippet}")
                
                if len(topic_messages) > 6:
                    lines.append(f"   ... та ще {len(topic_messages) - 6} повідомлень")
        elif topic_data:
            # Fallback for old format
            snippet_texts = [snippet if isinstance(snippet, str) else snippet[1] for snippet in topic_data[:4]]
            topic_summary = ' '.join(snippet_texts)
            lines.append(f"• <b>{topic.capitalize()}</b>: {topic_summary}")
    
    return "\n".join(lines) if lines else ""
