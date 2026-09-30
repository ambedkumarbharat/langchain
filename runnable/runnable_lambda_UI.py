import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)


# Keep the existing environment and LangChain setup unchanged.
load_dotenv()

prompt1 = ChatPromptTemplate.from_messages([
    ("system", """
you are just key point provider
"""),
    ("human", "{user_input}"),
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", " you are content explainer wrtitter on key "),
    ("human", "{key}"),
])

llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free",
)

parser = StrOutputParser()


def token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))


sequential_chain = RunnableSequence(prompt1, llm, parser)
parallel_chain = RunnableParallel({
    "key": RunnablePassthrough(),
    "content": RunnableSequence(prompt2, llm, parser),
    "token_count": RunnableLambda(token_count),
})

chain = RunnableSequence(sequential_chain, parallel_chain)


st.set_page_config(
    page_title="AI Topic Explainer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 880px; padding-top: 2.5rem; }
        .hero { text-align: center; padding: 0.3rem 0 1.5rem; }
        .hero h1 { margin-bottom: 0.35rem; }
        .hero p { color: #6b7280; margin: 0; }
        [data-testid="stChatMessage"] { border-radius: 16px; }
        .result-card {
            background: linear-gradient(135deg, #f8fbff 0%, #f5f3ff 100%);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.65rem 0;
        }
        .result-card h4 { margin: 0 0 0.55rem; }
        .word-count {
            display: inline-block;
            background: #ede9fe;
            color: #5b21b6;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🧠 AI Topic Explainer</h1>
        <p>Ask about any topic to get key points and a clear explanation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def show_result(result: dict) -> None:
    """Render one chain response in the chat without changing its data."""
    with st.container(border=True):
        st.markdown("#### 🔑 Key")
        st.markdown(result["key"])

    with st.container(border=True):
        st.markdown("#### 📖 Explanation")
        st.markdown(result["content"])

    st.markdown(
        f"<span class='word-count'>📊 Token Count: {result['token_count']}</span>",
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_result(message["content"])

if user_data := st.chat_input("Ask about any topic..."):
    st.session_state.messages.append({"role": "user", "content": user_data})

    with st.chat_message("user"):
        st.markdown(user_data)

    with st.chat_message("assistant"):
        with st.spinner("Creating key points and explanation..."):
            try:
                result = chain.invoke({"user_input": user_data})
            except Exception as error:
                st.error("I couldn't generate a response. Check your .env API setup and try again.")
                st.caption(f"Details: {error}")
            else:
                show_result(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
