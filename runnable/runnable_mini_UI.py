

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://api.xkiro.com/v1"
DEFAULT_MODEL = "qwen/qwen3.8-omni-flash:free"


@dataclass
class StageUsage:
    """Tokens used by one LLM call, and whether the provider reported them."""

    stage: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    source: str


def _estimate_tokens(text: str) -> tuple[int, str]:
    """Return a clearly labelled estimate when the provider gives no usage data."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text)), "estimated (cl100k_base)"
    except (ImportError, ValueError):
        # Kept as a last-resort estimate so the UI still works without tiktoken.
        return max(1, (len(text) + 3) // 4), "estimated (characters / 4)"


def _message_text(value: Any) -> str:
    """Convert LangChain prompt values and messages into text for estimation."""
    if hasattr(value, "to_messages"):
        value = value.to_messages()
    if isinstance(value, list):
        return "\n".join(_message_text(item) for item in value)
    if isinstance(value, BaseMessage):
        return str(value.content)
    return str(value)


def _usage_from_response(response: Any) -> dict[str, Any]:
    """Support LangChain's standard and OpenAI-compatible usage locations."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return usage

    metadata = getattr(response, "response_metadata", {}) or {}
    for key in ("token_usage", "usage"):
        candidate = metadata.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _pick_usage(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def call_stage(
    *, stage: str, prompt: PromptTemplate | ChatPromptTemplate, values: dict[str, str], llm: ChatOpenAI
) -> tuple[str, StageUsage]:
    """Run one prompt + model call and capture actual usage when it is returned."""
    rendered_prompt = prompt.invoke(values)
    response = llm.invoke(rendered_prompt)
    output = StrOutputParser().invoke(response)

    provider_usage = _usage_from_response(response)
    input_tokens = _pick_usage(provider_usage, "input_tokens", "prompt_tokens")
    output_tokens = _pick_usage(provider_usage, "output_tokens", "completion_tokens")
    total_tokens = _pick_usage(provider_usage, "total_tokens")
    has_provider_input = input_tokens is not None
    has_provider_output = output_tokens is not None

    estimated_input, estimate_source = _estimate_tokens(_message_text(rendered_prompt))
    estimated_output, _ = _estimate_tokens(output)

    # A stage is called "provider reported" only when both sides of the call are known.
    # Missing fields are still useful, but are explicitly filled with estimates.
    if input_tokens is None:
        input_tokens = estimated_input
    if output_tokens is None:
        output_tokens = estimated_output
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    if has_provider_input and has_provider_output:
        source = "provider reported"
    elif not has_provider_input and not has_provider_output:
        source = estimate_source
    else:
        source = "mixed: provider + estimate"

    return output, StageUsage(stage, input_tokens, output_tokens, total_tokens, source)


# Prompt 1: turn a casual request into a structured learning instruction.
prompt1 = PromptTemplate(
    template="""
You are an expert learning-prompt generator.

The user will give you a simple request about something they want to learn.
Convert it into a clear, detailed, structured instruction that another AI tutor can use.

The instruction must identify the topic and learning goal, assume beginner level unless
specified otherwise, cover concepts in a logical basic-to-advanced order, include practical
examples and exercises where appropriate, and focus on understanding rather than memorization.
Do not answer the user's question. Output only the final learning instruction.

User request:
{user_input}
""",
    input_variables=["user_input"],
)

# Prompt 2: pull out the outline/key points from Prompt 1's instruction.
prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert learning-content organizer. Identify the most important
concepts from the learning instruction. Return concise, beginner-friendly key points in a
logical learning order. Do not explain them in detail, ask questions, or add unrelated topics.
Output only the key points.""",
        ),
        ("human", "{learning_prompt}"),
    ]
)

# Prompt 3 remains the teaching stage in the final RunnableParallel branch.
prompt3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI tutor. Teach the learner using the provided learning
instruction and key points. Explain in simple language, follow the points in order, move from
basic to advanced, use practical examples where useful, and focus on understanding. Do not ask
questions unless the learner specifically requests them.""",
        ),
        (
            "human",
            """Learning Instruction:
{learning_prompt}

Key Points:
{key_points}

Now explain the topic to the learner.""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_llm(base_url: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(base_url=base_url, model=model, temperature= 0.4)


def build_chain(llm: ChatOpenAI) -> RunnableSequence:
    """Keep the original sequence → sequence → RunnableParallel learning architecture."""

    def run_prompt1(values: dict[str, str]) -> dict[str, Any]:
        learning_prompt, usage = call_stage(stage="Prompt 1", prompt=prompt1, values=values, llm=llm)
        return {"learning_prompt": learning_prompt, "usage": [usage]}

    def run_prompt2(values: dict[str, Any]) -> dict[str, Any]:
        key_points, usage = call_stage(
            stage="Prompt 2", prompt=prompt2, values={"learning_prompt": values["learning_prompt"]}, llm=llm
        )
        return {
            "learning_prompt": values["learning_prompt"],
            "key_points": key_points,
            "usage": [*values["usage"], usage],
        }

    def run_prompt3(values: dict[str, Any]) -> dict[str, Any]:
        content, usage = call_stage(
            stage="Prompt 3",
            prompt=prompt3,
            values={"learning_prompt": values["learning_prompt"], "key_points": values["key_points"]},
            llm=llm,
        )
        return {"content": content, "prompt3_usage": usage}

    # The passthroughs make each named stage a real RunnableSequence while preserving
    # the value produced by its tracking wrapper.
    sequential_chain_1 = RunnableSequence(RunnableLambda(run_prompt1), RunnablePassthrough())
    sequential_chain_2 = RunnableSequence(RunnableLambda(run_prompt2), RunnablePassthrough())
    parallel_chain = RunnableParallel(
        {
            "key": RunnableLambda(lambda values: values["key_points"]),
            "content_result": RunnableLambda(run_prompt3),
            "state": RunnablePassthrough(),
        }
    )

    return RunnableSequence(sequential_chain_1, sequential_chain_2, parallel_chain)


def run_learning_request(user_input: str, llm: ChatOpenAI) -> dict[str, Any]:
    result = build_chain(llm).invoke({"user_input": user_input})
    content_result = result["content_result"]
    usage: list[StageUsage] = [*result["state"]["usage"], content_result["prompt3_usage"]]
    return {"key_points": result["key"], "content": content_result["content"], "usage": usage}


def usage_totals(usage: list[StageUsage]) -> dict[str, int]:
    return {
        "input": sum(item.input_tokens for item in usage),
        "output": sum(item.output_tokens for item in usage),
        "total": sum(item.total_tokens for item in usage),
    }


def is_estimated(usage: list[StageUsage]) -> bool:
    return any(item.source != "provider reported" for item in usage)


def render_usage(usage: list[StageUsage], *, title: str = "Token usage") -> None:
    totals = usage_totals(usage)
    estimated = is_estimated(usage)
    st.caption(f"{title} · {'includes estimates' if estimated else 'provider reported'}")
    input_col, output_col, total_col = st.columns(3)
    input_col.metric("Input", f"{totals['input']:,}")
    output_col.metric("Output", f"{totals['output']:,}")
    total_col.metric("Total", f"{totals['total']:,}")

    with st.expander("Stage-by-stage token details"):
        st.dataframe(
            [asdict(item) for item in usage],
            column_config={
                "stage": "Stage",
                "input_tokens": "Input",
                "output_tokens": "Output",
                "total_tokens": "Total",
                "source": "Count source",
            },
            hide_index=True,
            use_container_width=True,
        )


def session_usage(messages: list[dict[str, Any]]) -> list[StageUsage]:
    return [stage for message in messages for stage in message["usage"]]


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="Learning Guide", page_icon="🧠", layout="centered")
    init_session()

    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    with st.sidebar:
        st.header("Learning Guide")
        st.caption("Three-stage LangChain tutor")
        st.caption(f"Model: `{model}`")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            st.divider()
            render_usage(session_usage(st.session_state.messages), title="Session usage")

    st.title("🧠 Learning Guide")
    st.caption("Ask what you want to learn. Each answer is created by Prompt 1 → Prompt 2 → Prompt 3.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Add `OPENAI_API_KEY` to a `.env` file before sending a message.")

    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(message["user_input"])
        with st.chat_message("assistant"):
            st.markdown("#### Key points")
            st.markdown(message["key_points"])
            st.markdown("#### Explanation")
            st.markdown(message["content"])
            render_usage(message["usage"])

    user_input = st.chat_input("What would you like to learn?")
    if not user_input:
        return
    if not os.getenv("OPENAI_API_KEY"):
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Building your learning guide…"):
            try:
                response = run_learning_request(user_input, get_llm(base_url, model))
            except Exception as error:
                st.error(f"The learning chain could not run: {error}")
                return
        st.markdown("#### Key points")
        st.markdown(response["key_points"])
        st.markdown("#### Explanation")
        st.markdown(response["content"])
        render_usage(response["usage"])

    st.session_state.messages.append({"user_input": user_input, **response})
    st.rerun()


if __name__ == "__main__":
    main()
