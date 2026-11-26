"""Development conversations for generation review; no response lookup rules."""

from robbie.core.conversation.model_chat import ModelChat


BASIC_SCENARIOS = [
    ["Hi, how are you?"],
    ["I have a free afternoon. Suggest something simple to do.", "I would rather stay indoors. Any other idea?"],
    ["What makes a good friend?", "Please say that in one short sentence."],
    ["I like drawing but I am a beginner. How can I get started?"],
]


def basic_samples(model, tokenizer, max_tokens=160):
    chat = ModelChat(model, tokenizer, max_tokens=max_tokens, temperature=0,
                     assistant_eos=getattr(model, "dialogue_format", "legacy") == "assistant-eos-v1",
                     repetition_penalty=1.0)
    rows = []
    was_training = model.training
    try:
        for index, questions in enumerate(BASIC_SCENARIOS):
            chat.messages.clear()
            for question in questions:
                answer = chat.reply(question)
                rows.append({"scenario": index, "question": question, "answer": answer,
                             **chat.last_generation})
        return rows
    finally:
        model.train(was_training)
