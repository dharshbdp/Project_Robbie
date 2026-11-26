"""Unit tests verifying Hallucination Guard, Gibberish Interception, and Web Search Confirmation."""

from unittest.mock import MagicMock, patch
import pytest

from robbie.core.conversation.engine import ChatEngine
from robbie.core.tokenizer.char_tokenizer import CharTokenizer


def create_engine():
    tok = CharTokenizer()
    tok.train(["User: Robbie: \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?."])
    model = MagicMock()
    model.config.block_size = 256
    engine = ChatEngine(model, tok, device="cpu")
    return engine


class TestHallucinationAndGibberishGuard:
    """Ensure Robbie never emits repetitive hallucinations or garbage answers."""

    def test_detection_of_repetitive_gibberish(self):
        samples_gibberish = [
            "A good day with a good day and time to shop the park is a great time.",
            "Many is a good day to start with a good day, so I'm here to help you think about.",
            "A good day can be a good day and set of parking. It's important to consider your day and going for the day.",
            "The Curm, Cosaurus Proper CE is a special flower proof C.",
            "time directly resulting in the same list",
        ]
        for s in samples_gibberish:
            assert ChatEngine.is_hallucination_or_gibberish(s) is True, f"Failed to detect gibberish: {s}"

    def test_clean_answers_pass_detection(self):
        clean_samples = [
            "Python is a high-level, general-purpose programming language known for its clear syntax and readability.",
            "Hi there! I am Robbie, your personal artificial intelligence assistant. What would you like to work on today?",
            "RAM stands for Random Access Memory, which is fast temporary system memory.",
            "It is 10:28:15 on 17 October 2026 (UTC+05:30).",
        ]
        for s in clean_samples:
            assert ChatEngine.is_hallucination_or_gibberish(s) is False, f"False positive on clean text: {s}"

    def test_intercepts_gibberish_and_offers_web_search(self):
        chat = create_engine()
        # Simulate model degenerating into repetitive word-salad
        chat.model.generate.return_value = MagicMock()
        chat.tokenizer.decode = MagicMock(
            return_value="A good day with a good day and time to shop the park is a great time."
        )

        reply = chat.generate_reply("Who invented the saxophone?")
        assert "I don't know the answer to that from memory" in reply
        assert "Should I look it up on the internet, Dharsh?" in reply
        assert chat.pending_search_query == "Who invented the saxophone?"

    def test_confirming_search_offer_executes_web_research(self):
        chat = create_engine()
        chat.pending_search_query = "who invented the saxophone"

        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "The saxophone was invented by the Belgian instrument maker Adolphe Sax in the early 1840s.\n\n"
                "**Sources:**\n- [Saxophone Wikipedia](https://en.wikipedia.org/wiki/Saxophone)"
            )
            reply = chat.respond("yes please")
            assert "Adolphe Sax" in reply
            mock_ra.answer_query.assert_called_once_with("who invented the saxophone")
            assert chat.pending_search_query is None

    def test_declining_search_offer_resets_gracefully(self):
        chat = create_engine()
        chat.pending_search_query = "who invented the saxophone"

        reply = chat.respond("no thanks")
        assert "Understood" in reply
        assert chat.pending_search_query is None

    def test_auto_routes_open_domain_knowledge_queries_to_web(self):
        chat = create_engine()
        open_domain_queries = [
            "what is quantum computing",
            "can dogs eat chocolate",
            "who was marie curie",
            "how do batteries work",
            "where is mount everest",
            "tell me about black holes",
        ]
        for q in open_domain_queries:
            assert chat.is_web_knowledge_query(q) is True, f"Failed to route open-domain query: {q}"

    def test_detects_word_salad_and_stacked_verb_hallucination(self):
        gibberish_cases = [
            "Arix, Kiday (L)",
            "Voliders are the world is a normal ness and all the networks of the networks",
            "curm properly resulting in the world",
            "trainate yespecial",
        ]
        for s in gibberish_cases:
            assert ChatEngine.is_hallucination_or_gibberish(s) is True, f"Failed to detect: {s}"

    def test_list_and_creator_queries_classified_as_web_knowledge(self):
        chat = create_engine()
        list_queries = [
            "name 10 nolan movies",
            "give 10 nolan movies",
            "list 10 nolan movies",
            "christopher nolan filmography",
            "fromsoftware games",
            "5 movies by spielberg",
        ]
        for q in list_queries:
            assert chat.is_web_knowledge_query(q) is True, f"Expected web knowledge query: {q}"

    def test_name_nolan_movies_routes_to_answer_query(self):
        chat = create_engine()
        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "Here are 10 notable films directed by Christopher Nolan:\n\n"
                "1. **Following** (1998) - Nolan's feature debut.\n"
                "2. **Memento** (2000) - Psychological thriller.\n"
                "3. **Batman Begins** (2005) - Reboot of the Batman franchise."
            )
            reply = chat.respond("name 10 nolan movies")
            assert "Following" in reply
            assert "Memento" in reply
            mock_ra.answer_query.assert_called_once_with("name 10 nolan movies")
            assert chat.context.messages[-1].content == reply

    def test_research_and_give_cleans_topic_and_calls_answer_query(self):
        chat = create_engine()
        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "Here are 10 notable films directed by Christopher Nolan:\n\n"
                "1. **Following** (1998)\n"
                "2. **Memento** (2000)"
            )
            reply = chat.respond("Research and give 10 nolan movies")
            assert "Following" in reply
            mock_ra.answer_query.assert_called_once_with("10 nolan movies")

    def test_conversational_repair_clarifies_previous_web_query(self):
        chat = create_engine()
        chat.context.add_user_message("name 10 nolan movies")
        chat.context.add_assistant_message("I am listening! How can I assist you?")

        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = "1. Following (1998)\n2. Memento (2000)"

            reply = chat.respond("what?")
            assert "Let me clarify that with accurate information" in reply
            assert "Following (1998)" in reply
            mock_ra.answer_query.assert_called_once_with("name 10 nolan movies")

    def test_multi_turn_name_query_does_not_hijack_following_list_query(self):
        chat = create_engine()
        # Turn 1: User asked for Fincher movies
        chat.context.add_user_message("Name 10 David Fincher movies")
        chat.context.add_assistant_message("Here are 10 movies directed by David Fincher: 1. Seven (1995)")
        # Turn 2: User asks for recommendations
        chat.context.add_user_message("Which one would you recommend?")
        chat.context.add_assistant_message("I would recommend Seven or Zodiac.")

        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "Here are 5 notable video games directed by Hideo Kojima:\n\n"
                "1. **Metal Gear** (1987)\n"
                "2. **Snatcher** (1988)\n"
                "3. **Metal Gear Solid** (1998)"
            )
            # Turn 3: User says 'name 5 kojima games' - must NOT return 'David Fincher.'!
            reply = chat.respond("name 5 kojima games")
            assert "David Fincher." not in reply
            assert "Metal Gear" in reply
            mock_ra.answer_query.assert_called_once_with("name 5 kojima games")

    def test_detects_broken_english_and_dangling_fragments(self):
        broken_sample = (
            "I'm sorry, but I don't have been not exist you. My name is the give provide "
            "a few time of our got and I can also be able. If you want to useful something "
            "about a new book a hobby a good park with an cands that are a great way to make items and creative on a"
        )
        assert ChatEngine.is_hallucination_or_gibberish(broken_sample) is True

    def test_contextual_recommendation_from_previous_list(self):
        chat = create_engine()
        chat.context.add_user_message("Name 10 David Fincher movies")
        chat.context.add_assistant_message(
            "Here are 10 notable movies directed by **David Fincher** in chronological order:\n\n"
            "1. **Seven** (1995) - Acclaimed crime thriller.\n"
            "2. **The Game** (1997) - Psychological mystery thriller.\n"
            "3. **Fight Club** (1999) - Iconic dark drama.\n"
            "4. **Panic Room** (2002) - Home invasion thriller.\n"
            "5. **Zodiac** (2007) - Atmospheric procedural thriller."
        )

        reply = chat.respond("Which one would you recommend for a person who like thrillers?")
        assert ("Seven" in reply or "Zodiac" in reply or "The Game" in reply)
        assert "recommend" in reply.lower()
        assert "not exist you" not in reply
        assert ChatEngine.is_hallucination_or_gibberish(reply) is False


