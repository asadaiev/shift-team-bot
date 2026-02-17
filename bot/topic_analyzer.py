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

# Track if OpenAI quota was exceeded to avoid repeated failed requests
_openai_quota_exceeded = False

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
    'м', 'т', 'й', 'ж', 'б', 'ж', 'то', 'а', 'ну', 'о', 'у', 'е', 'и', 'о', 'а',
    # Add more stop words for better topic detection
    'тебе', 'мене', 'його', 'її', 'нас', 'вас', 'їх',
    'просто', 'думаю', 'ніхуя', 'нічого', 'ніколи', 'ніде', 'нікуди',
    'може', 'можна', 'треба', 'потрібно', 'варто',
    'було', 'буде', 'є', 'був', 'була', 'були',
    'щось', 'хтось', 'десь', 'кудись', 'звідкись',
    'вже', 'ще', 'тільки', 'лише', 'навіть', 'також'
}

# Minimum word length to consider (increased to filter out short words)
MIN_WORD_LENGTH = 4

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
    global _openai_quota_exceeded
    logger.info(f"generate_text_summary: checking OpenAI - USE_OPENAI_SUMMARY={Config.USE_OPENAI_SUMMARY}, OPENAI_AVAILABLE={OPENAI_AVAILABLE}, has_key={bool(Config.OPENAI_API_KEY)}, quota_exceeded={_openai_quota_exceeded}")
    if Config.USE_OPENAI_SUMMARY and OPENAI_AVAILABLE and Config.OPENAI_API_KEY and not _openai_quota_exceeded:
        try:
            logger.info("generate_text_summary: trying OpenAI")
            result = await generate_openai_detailed_summary(all_text, messages)
            if result:
                logger.info(f"generate_text_summary: OpenAI success, length: {len(result)}")
                return result
            else:
                logger.warning("generate_text_summary: OpenAI returned empty result")
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}, falling back to sumy", exc_info=True)
    else:
        logger.warning(f"generate_text_summary: OpenAI not available - USE_OPENAI_SUMMARY={Config.USE_OPENAI_SUMMARY}, OPENAI_AVAILABLE={OPENAI_AVAILABLE}, has_key={bool(Config.OPENAI_API_KEY)}")
    
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
        return await generate_simple_fallback_summary(messages)
    
    logger.warning("generate_text_summary: all methods failed, using simple fallback")
    return await generate_simple_fallback_summary(messages)


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


async def generate_openai_detailed_summary(text: str, messages: List[Tuple[str, str, str]] = None) -> str:
    """Generate detailed summary by topics using OpenAI API."""
    global _openai_quota_exceeded
    if not OPENAI_AVAILABLE or not Config.OPENAI_API_KEY or _openai_quota_exceeded:
        return ""
    
    # If we have messages, use the same logic as generate_simple_fallback_summary to get topic names
    if messages:
        topics = analyze_topics(messages)
        if topics:
            topic_groups = group_messages_by_topic(messages, topics)
            # Use OpenAI to generate narratives for each topic
            result_lines = []
            topic_index = 0
            for topic, topic_data in list(topic_groups.items())[:8]:
                if isinstance(topic_data, tuple) and len(topic_data) == 2:
                    first_mention, topic_messages = topic_data
                    if topic_messages:
                        topic_index += 1
                        participants = set(username for username, _ in topic_messages)
                        participant_count = len(participants)
                        
                        # Get better topic name using OpenAI
                        better_topic_name = None
                        if not _openai_quota_exceeded:
                            try:
                                better_topic_name = await generate_topic_name(topic_messages)
                            except Exception as e:
                                logger.warning(f"Failed to generate topic name: {e}")
                        
                        display_topic = better_topic_name if better_topic_name else topic.capitalize()
                        
                        # Get all messages for this topic
                        all_topic_texts = [msg_text for _, msg_text in topic_messages]
                        topic_text = '\n'.join(all_topic_texts)
                        
                        # Generate narrative using OpenAI
                        narrative_text = None
                        if not _openai_quota_exceeded:
                            try:
                                narrative_text = await generate_topic_narrative(display_topic, topic_text, first_mention, participant_count)
                            except Exception as e:
                                logger.warning(f"Failed to generate OpenAI narrative: {e}")
                        
                        # Fallback: create simple narrative
                        if not narrative_text:
                            meaningful_messages = [msg.strip() for msg in all_topic_texts if len(msg.strip()) > 15]
                            if meaningful_messages:
                                narrative_text = ' '.join(meaningful_messages[:5])
                                narrative_text = ' '.join(narrative_text.split())
                                narrative_text = re.sub(r'[-•*]\s+', ' ', narrative_text)
                                narrative_text = re.sub(r'\d+\.\s+', ' ', narrative_text)
                                narrative_text = re.sub(r'[a-z]\)\s+', ' ', narrative_text)
                                if len(narrative_text) > 500:
                                    narrative_text = narrative_text[:500] + '...'
                        
                        if narrative_text:
                            # Ensure narrative starts with "В цій темі обговорювали..."
                            narrative_lower = narrative_text.lower().strip()
                            if not any(narrative_lower.startswith(prefix) for prefix in ('в цій темі', 'в цій', 'обговорювали', 'говорили', 'розмовляли')):
                                narrative_text = f"В цій темі обговорювали {display_topic.lower()}. {narrative_text}"
                            
                            result_lines.append(f"{topic_index}. <b>{display_topic}</b> (підняв: <b>{first_mention}</b>, учасників: {participant_count})")
                            result_lines.append(f"   {narrative_text}")
                            
                            if len(topic_messages) > 5:
                                result_lines.append(f"   <i>... та ще {len(topic_messages) - 5} повідомлень про цю тему</i>")
                            
                            # Add spacing between topics (except after the last one)
                            result_lines.append("")
                            result_lines.append("")
            
            if result_lines:
                # Remove trailing empty lines
                while result_lines and result_lines[-1] == "":
                    result_lines.pop()
                return "\n".join(result_lines)
    
    # Fallback to old method if no messages provided
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
                    "content": "Ти допомагаєш створювати живі та змістовні summary обговорень українською мовою. Створюй структурований звіт по темах, де кожна тема описується одним зв'язним текстом без підпунктів (максимум 10 речень на тему, краще 5-7). Пиши живо та конкретно: 'В цій темі обговорювали таке-то і таке-то', згадуй конкретні деталі що саме говорилось, а не просто загальний переклад. Кожна тема має бути одним оповідаючим текстом з конкретними деталями."
                },
                {
                    "role": "user", 
                    "content": f"Проаналізуй наступне обговорення та створи summary українською мовою в ЖИВОМУ ОПОВІДАЮЧОМУ форматі. Організуй його по темах, де кожна тема - це ОДИН ЗВ'ЯЗНИЙ ТЕКСТ без підпунктів, списків, маркерів (-, •, 1., 2., a), b) тощо). Максимум 10 речень на тему (краще 5-7). Пиши живо та конкретно: 'В цій темі обговорювали таке-то і таке-то', згадуй конкретні деталі що саме говорилось, конкретні приклади з обговорення, а не просто загальний переклад.\n\nВАЖЛИВО: Пиши живо та конкретно. Наприклад:\n- 'В цій темі обговорювали змії, які трапляються на різних континентах'\n- 'Обговорювали нові оновлення в CS2, зокрема зміни в механіці стрільби'\n- 'Говорили про рейтингову систему Faceit та як підвищити свій Elo'\n\nЗАБОРОНЕНО використовувати:\n- Маркери (-, •, *)\n- Нумеровані списки (1., 2., 3.)\n- Літерні списки (a), b), c))\n- Будь-які інші підпункти\n\nПРАВИЛЬНИЙ формат (один зв'язний текст для кожної теми):\n🎯 Тема 1: В цій темі обговорювали [конкретна тема]. [Конкретні деталі що саме говорилось]. [Конкретні приклади з обговорення]. [Висновки або рішення]. Все одним текстом без розбиття на пункти.\n🎯 Тема 2: В цій темі обговорювали [інша тема]. [Конкретні деталі]. Також одним текстом.\n\nОбговорення:\n\n{text}"
                }
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        # Remove any duplicate topic headers that might be in the text
        summary = re.sub(r'🎯\s*[^(]*\([^)]*\)[^.]*\.?\s*', '', summary)
        summary = re.sub(r'🎯\s*[^:]*:\s*', '', summary)
        
        # Remove bullet points and numbered lists within topics - more aggressive cleaning
        # Remove bullet points (-, •, *, etc.)
        summary = re.sub(r'\n\s*[-•*]\s+', ' ', summary)
        summary = re.sub(r'[-•*]\s+', ' ', summary)
        # Remove numbered lists (1., 2., etc.)
        summary = re.sub(r'\n\s*\d+\.\s+', ' ', summary)
        summary = re.sub(r'\d+\.\s+', ' ', summary)
        # Remove any remaining list markers
        summary = re.sub(r'\n\s*[a-z]\)\s+', ' ', summary)  # a), b), c)
        summary = re.sub(r'[a-z]\)\s+', ' ', summary)
        
        # Split by topic headers and clean each topic
        topics = re.split(r'🎯\s*[^:]+:', summary)
        cleaned_topics = []
        for topic in topics:
            if topic.strip():
                # Remove all list markers from topic text
                topic = re.sub(r'[-•*]\s+', ' ', topic)
                topic = re.sub(r'\d+\.\s+', ' ', topic)
                topic = re.sub(r'[a-z]\)\s+', ' ', topic)
                # Replace multiple newlines with single space
                topic = re.sub(r'\n+', ' ', topic)
                # Clean up multiple spaces
                topic = ' '.join(topic.split())
                if topic.strip():
                    cleaned_topics.append(topic.strip())
        
        # Reconstruct summary with cleaned topics
        if cleaned_topics:
            result_lines = []
            topic_headers = re.findall(r'🎯\s*[^:]+:', summary)
            for i, header in enumerate(topic_headers):
                if i < len(cleaned_topics):
                    result_lines.append(f"{header.strip()} {cleaned_topics[i]}")
            summary = '\n\n'.join(result_lines) if result_lines else summary
        else:
            # Fallback: just clean the original text
            summary = re.sub(r'\n+', ' ', summary)
            summary = ' '.join(summary.split())
        
        return summary.strip()
    except Exception as e:
        error_str = str(e).lower()
        # Check for quota exceeded (429) or insufficient quota
        if '429' in error_str or 'insufficient_quota' in error_str or 'quota' in error_str:
            logger.error(f"OpenAI quota exceeded, disabling OpenAI for this session: {e}")
            # Mark quota as exceeded to avoid repeated failed requests
            _openai_quota_exceeded = True
        else:
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


async def generate_topic_name(topic_messages: List[Tuple[str, str]]) -> Optional[str]:
    """Generate a better topic name using OpenAI based on message content."""
    global _openai_quota_exceeded
    if not Config.USE_OPENAI_SUMMARY or not OPENAI_AVAILABLE or not Config.OPENAI_API_KEY or _openai_quota_exceeded:
        return None
    
    if not topic_messages:
        return None
    
    try:
        client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Get all messages for this topic
        all_texts = [msg_text for _, msg_text in topic_messages[:10]]  # Use first 10 messages
        topic_text = '\n'.join(all_texts)
        
        # Truncate if too long
        if len(topic_text) > 1500:
            topic_text = topic_text[:1500] + "..."
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ти допомагаєш генерувати короткі, інформативні назви тем обговорень українською мовою. Назва має бути 1-3 слова, що точно описують про що йдеться в обговоренні."
                },
                {
                    "role": "user",
                    "content": f"Проаналізуй наступні повідомлення та створи коротку інформативну назву теми (1-3 слова) українською мовою, що точно описує про що йдеться. Назва має бути конкретною та змістовною, не просто перше слово з повідомлення.\n\nПовідомлення:\n{topic_text}\n\nНазва теми (тільки назва, без додаткового тексту):"
                }
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        topic_name = response.choices[0].message.content.strip()
        # Remove quotes if present
        topic_name = topic_name.strip('"\'')
        return topic_name if topic_name else None
    except Exception as e:
        error_str = str(e).lower()
        # Check for quota exceeded (429) or insufficient quota
        if '429' in error_str or 'insufficient_quota' in error_str or 'quota' in error_str:
            logger.warning(f"OpenAI quota exceeded, skipping topic name generation: {e}")
            _openai_quota_exceeded = True
        else:
            logger.warning(f"Error generating topic name: {e}")
        return None


async def generate_simple_fallback_summary(messages: List[Tuple[str, str, str]]) -> str:
    """Generate a detailed topic-based summary. Uses OpenAI for better narrative if available."""
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
    
    # Generate detailed summary by topics in narrative format
    lines = []
    topic_index = 0
    for topic, topic_data in list(topic_groups.items())[:8]:  # Top 8 topics for more detail
        topic_index += 1
        if isinstance(topic_data, tuple) and len(topic_data) == 2:
            first_mention, topic_messages = topic_data
            if topic_messages:
                # Count unique participants
                participants = set(username for username, _ in topic_messages)
                participant_count = len(participants)
                
                # Try to generate better topic name using OpenAI
                better_topic_name = None
                global _openai_quota_exceeded
                if Config.USE_OPENAI_SUMMARY and OPENAI_AVAILABLE and Config.OPENAI_API_KEY and not _openai_quota_exceeded:
                    try:
                        better_topic_name = await generate_topic_name(topic_messages)
                    except Exception as e:
                        logger.warning(f"Failed to generate topic name: {e}")
                
                # Use better name if available, otherwise use original
                display_topic = better_topic_name if better_topic_name else topic.capitalize()
                
                # Get all messages for this topic
                all_topic_texts = [msg_text for _, msg_text in topic_messages]
                topic_text = '\n'.join(all_topic_texts)
                
                # Try to use OpenAI for better narrative if available
                narrative_text = None
                if Config.USE_OPENAI_SUMMARY and OPENAI_AVAILABLE and Config.OPENAI_API_KEY and not _openai_quota_exceeded:
                    try:
                        narrative_text = await generate_topic_narrative(display_topic, topic_text, first_mention, participant_count)
                    except Exception as e:
                        logger.warning(f"Failed to generate OpenAI narrative for topic {display_topic}: {e}")
                
                # Fallback: create simple narrative from messages
                if not narrative_text:
                    meaningful_messages = [msg.strip() for msg in all_topic_texts if len(msg.strip()) > 15]
                    if meaningful_messages:
                        # Take first 3-5 meaningful messages
                        selected_messages = meaningful_messages[:5]
                        # Try to create a more coherent narrative
                        narrative_text = ' '.join(selected_messages)
                        # Clean up: remove excessive punctuation and spaces
                        narrative_text = ' '.join(narrative_text.split())
                        # Remove any topic headers that might be in the text (like "🎯 Тема:" or "🎯 Тема (підняв: ...)")
                        narrative_text = re.sub(r'🎯\s*[^(]*\([^)]*\)[^.]*\.?\s*', '', narrative_text)
                        narrative_text = re.sub(r'🎯\s*[^:]*:\s*', '', narrative_text)
                        narrative_text = narrative_text.strip()
                        # Limit to 400-500 chars for readability
                        if len(narrative_text) > 500:
                            narrative_text = narrative_text[:500] + '...'
                
                if narrative_text:
                    # Ensure narrative starts with "В цій темі обговорювали..." if it doesn't already
                    narrative_lower = narrative_text.lower().strip()
                    if not any(narrative_lower.startswith(prefix) for prefix in ('в цій темі', 'в цій', 'обговорювали', 'говорили', 'розмовляли')):
                        narrative_text = f"В цій темі обговорювали {display_topic.lower()}. {narrative_text}"
                    
                    lines.append(f"")
                    lines.append(f"{topic_index}. <b>{display_topic}</b> (підняв: <b>{first_mention}</b>, учасників: {participant_count})")
                    lines.append(f"   {narrative_text}")
                    
                    if len(topic_messages) > 5:
                        lines.append(f"   <i>... та ще {len(topic_messages) - 5} повідомлень про цю тему</i>")
                    
                    # Add spacing between topics (except after the last one)
                    lines.append("")
                    lines.append("")
        elif topic_data:
            # Fallback for old format
            snippet_texts = [snippet if isinstance(snippet, str) else snippet[1] for snippet in topic_data[:4]]
            topic_summary = ' '.join(snippet_texts)
            if len(topic_summary) > 400:
                topic_summary = topic_summary[:400] + '...'
            lines.append(f"🎯 <b>{topic.capitalize()}</b>: {topic_summary}")
    
    if lines:
        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)
    return ""


async def generate_topic_narrative(topic: str, topic_messages: str, first_mention: str, participant_count: int) -> Optional[str]:
    """Generate narrative summary for a specific topic using OpenAI."""
    global _openai_quota_exceeded
    if not OPENAI_AVAILABLE or not Config.OPENAI_API_KEY or _openai_quota_exceeded:
        return None
    
    try:
        client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Truncate if too long
        if len(topic_messages) > 2000:
            topic_messages = topic_messages[:2000] + "..."
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ти допомагаєш створювати оповідаючі summary обговорень українською мовою. Створюй зв'язний, оповідаючий текст що описує що обговорювалось, без згадок конкретних імен користувачів (якщо не критично важливо)."
                },
                {
                    "role": "user",
                    "content": f"Створи живий оповідаючий summary (максимум 10 речень, краще 5-7) про те, що обговорювалось на тему '{topic}'. Почни з 'В цій темі обговорювали [конкретна тема]' і далі опиши конкретні деталі що саме говорилось, конкретні приклади з обговорення, а не просто загальний переклад. Текст має бути зв'язним, оповідаючим, одним суцільним текстом БЕЗ підпунктів, списків, маркерів (-, •, 1., 2. тощо). Не згадуй імена користувачів, якщо це не критично важливо. Пиши живо та конкретно з деталями що саме обговорювалось.\n\nВАЖЛИВО: НЕ використовуй маркери, списки або підпункти. Тільки один зв'язний текст. Почни з 'В цій темі обговорювали...' і далі опиши конкретні деталі.\n\nОбговорення:\n{topic_messages}"
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        narrative = response.choices[0].message.content.strip()
        # Remove any topic headers that might be in the text (like "🎯 Тема:" or "🎯 Тема (підняв: ...)")
        narrative = re.sub(r'🎯\s*[^(]*\([^)]*\)[^.]*\.?\s*', '', narrative)
        narrative = re.sub(r'🎯\s*[^:]*:\s*', '', narrative)
        # Remove bullet points and numbered lists - aggressive cleaning
        narrative = re.sub(r'\n\s*[-•*]\s+', ' ', narrative)
        narrative = re.sub(r'[-•*]\s+', ' ', narrative)
        narrative = re.sub(r'\n\s*\d+\.\s+', ' ', narrative)
        narrative = re.sub(r'\d+\.\s+', ' ', narrative)
        narrative = re.sub(r'\n\s*[a-z]\)\s+', ' ', narrative)
        narrative = re.sub(r'[a-z]\)\s+', ' ', narrative)
        # Replace multiple newlines with single space
        narrative = re.sub(r'\n+', ' ', narrative)
        # Clean up multiple spaces
        narrative = ' '.join(narrative.split())
        return narrative.strip()
    except Exception as e:
        error_str = str(e).lower()
        # Check for quota exceeded (429) or insufficient quota
        if '429' in error_str or 'insufficient_quota' in error_str or 'quota' in error_str:
            logger.warning(f"OpenAI quota exceeded, skipping topic narrative: {e}")
            _openai_quota_exceeded = True
        else:
            logger.warning(f"Error generating topic narrative: {e}")
        return None
