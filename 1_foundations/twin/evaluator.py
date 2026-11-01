"""Guardrail: a second LLM call that checks each reply stays on professional topics."""

import anthropic
from pydantic import BaseModel, Field

from context import summary

# A different model family from the twin (OpenAI), so the check is independent.
# Swap to "claude-haiku-5-5" for a faster, cheaper check.
EVALUATOR_MODEL = "claude-opus-5-5"

# Reads ANTHROPIC_API_KEY from the environment (Render's Environment tab, or .env locally)
claude = anthropic.Anthropic()


class Evaluation(BaseModel):
    is_acceptable: bool = Field(description="True if the reply stays on professional topics, as defined in the rules")
    feedback: str = Field(description="If not acceptable, what is off-topic and how to fix it. Otherwise a short note.")


EVALUATOR_SYSTEM_PROMPT = f"""
You are a strict quality checker for an AI assistant that acts as a digital version of Bosi (Terry) Lin on his website.

Your only job: decide whether the assistant's latest reply stays on professional topics.

Acceptable replies talk about:
- Terry's career, background, education, skills, experience, projects and certifications
- the roles he is looking for, and recruiter screening topics: work authorization, sponsorship, location preferences,
  and how he approaches compensation
- professional opportunities, collaboration, or getting in touch (including asking for an email address)
- brief, polite greetings or small talk that lead back to professional topics
- politely declining an off-topic request and steering back to work
- saying he doesn't know something and that he will follow up
- a brief, light mention of the hobbies listed in his summary below, only when the user asks, and steering back to work

Unacceptable replies:
- engage with off-topic requests (recipes, sports, politics, news, general trivia, homework, writing unrelated code, etc.)
- go on at length about his personal life, or share personal details that are not in his summary below
  (family, relationships, health, home address, phone number, finances, opinions on non-work topics)
- drop the role of Terry's digital version

Judge the reply only. A user asking something off-topic is fine, as long as the reply does not go along with it.

Here is Terry's summary, so you know which personal details he is comfortable sharing:

{summary}
""".strip()


def text_of(content):
    """Gradio may give message content as a string or as a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def evaluator_user_prompt(reply, message, history):
    transcript = "\n".join(f"{h['role']}: {text_of(h['content'])}" for h in history) or "(no earlier messages)"
    return f"""Here is the conversation so far:

{transcript}

The user's latest message:
{message}

The assistant's reply to evaluate:
{reply}

Does this reply stay on professional topics? Give your verdict and feedback."""


def evaluate(reply, message, history) -> Evaluation:
    response = claude.beta.messages.parse(
        model=EVALUATOR_MODEL,
        max_tokens=4096,
        output_config={"effort": "low"},  # a quick yes/no judgment doesn't need deep thinking
        # If Claude's safety filters decline, the API retries on a fallback model automatically
        betas=["server-side-fallback-2026-07-01"],
        # Sent as a raw field so it also works on older SDKs (the course repo has anthropic 0.107)
        extra_body={"fallbacks": "default"},
        system=EVALUATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": evaluator_user_prompt(reply, message, history)}],
        output_format=Evaluation,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        # Raise so chat() falls back to showing the reply unchecked instead of crashing
        raise RuntimeError(f"Evaluator gave no verdict (stop_reason={response.stop_reason})")
    return response.parsed_output
