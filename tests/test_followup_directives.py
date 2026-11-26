"""Unit tests verifying contextual follow-up directives, anaphora resolution, and context retention."""

from unittest.mock import MagicMock, patch
import pytest

from robbie.core.conversation.engine import ChatEngine
from robbie.core.conversation.messages import Role, Message
from robbie.core.tokenizer.char_tokenizer import CharTokenizer


def create_engine():
    tok = CharTokenizer()
    tok.train(["User: Robbie: \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?."])
    model = MagicMock()
    model.config.block_size = 256
    return ChatEngine(model, tok, device="cpu")


class TestFollowupDirectives:
    """Ensure Robbie maintains multi-turn context and resolves follow-up directives."""

    def test_name_followup_extraction(self):
        chat = create_engine()
        # Seed previous exchange
        chat.context.add_user_message("Who invented the saxophone?")
        chat.context.add_assistant_message(
            "Here is what I found regarding Who invented the saxophone:\n\n"
            "- musician who invented the saxophone in the early 1840s, patenting it in 1846. "
            "- His son was Adolphe Sax, who invented the saxophone, the saxhorn and the saxotromba. "
            "- The Unkillable Weirdo Who Invented the Saxophone - Puppet History.\n\n"
            "**Sources:**\n- [Saxophone](https://en.wikipedia.org/?curid=26795)"
        )

        reply1 = chat.respond("just the name")
        assert "Adolphe Sax" in reply1

        reply2 = chat.respond("who?")
        assert "Adolphe Sax" in reply2

    def test_year_followup_extraction(self):
        chat = create_engine()
        chat.context.add_user_message("Who invented the saxophone?")
        chat.context.add_assistant_message(
            "- His son was Adolphe Sax, who invented the saxophone in the early 1840s, patenting it in 1846."
        )

        reply = chat.respond("what year?")
        assert "1846" in reply or "1840" in reply

    def test_pronoun_anaphora_resolution_to_web(self):
        chat = create_engine()
        chat.context.add_user_message("Who invented the saxophone?")
        chat.context.add_assistant_message(
            "- His son was Adolphe Sax, who invented the saxophone in the early 1840s, patenting it in 1846."
        )

        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "Adolphe Sax was born in Dinant, Belgium on 6 November 1814."
            )
            reply = chat.respond("where was he born?")
            assert "Dinant" in reply or "Belgium" in reply
            mock_ra.answer_query.assert_called_once_with("where was Adolphe Sax born?")

    def test_place_followup_extraction(self):
        chat = create_engine()
        chat.context.add_user_message("Where was Adolphe Sax born?")
        chat.context.add_assistant_message(
            "Antoine-Joseph Sax was born on 6 November 1814 in Dinant, in a post-Napoleonic polity of what is now Belgium."
        )

        reply1 = chat.respond("just the city")
        assert "Dinant" in reply1

        reply2 = chat.respond("where?")
        assert "Dinant" in reply2

    def test_elaboration_directive_routing(self):
        chat = create_engine()
        chat.context.add_user_message("Who invented the saxophone?")
        chat.context.add_assistant_message(
            "- His son was Adolphe Sax, who invented the saxophone in the early 1840s, patenting it in 1846."
        )

        with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
            mock_ra = mock_ra_cls.return_value
            mock_ra.answer_query.return_value = (
                "Adolphe Sax was a Belgian instrument maker who also invented the saxotromba and saxhorn."
            )
            reply = chat.respond("tell me more")
            assert "saxhorn" in reply
            mock_ra.answer_query.assert_called_once_with("Tell me more details about Adolphe Sax")

    def test_sources_stripped_in_context_prompt_to_prevent_token_choke(self):
        chat = create_engine()
        chat.context.add_user_message("Who invented the saxophone?")
        chat.context.add_assistant_message(
            "Adolphe Sax invented the saxophone.\n\n"
            "Sources: - [Saxophone](https://en.wikipedia.org/?curid=26795) - [Alto](https://en.wikipedia.org/?curid=556372)"
        )
        chat.context.add_user_message("Tell me more")

        prompt, ids = chat.context.get_context_prompt(assistant_eos=True)
        # Verify markdown URLs are not bloating the LLM prompt
        assert "https://" not in prompt
        assert "curid=" not in prompt
        assert "Who invented the saxophone?" in prompt
