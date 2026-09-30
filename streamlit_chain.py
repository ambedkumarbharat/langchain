import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel

from pydantic import BaseModel, Field
from typing import Literal


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="LangChain Chain Lab",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# MODELS
# ============================================================

groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7
)

openrouter_llm = ChatOpenRouter(
    model="nex-agi/nex-n2.5-mini:free"
)

parser = StrOutputParser()


# ============================================================
# COMMON UI HELPERS
# ============================================================

def show_step(number, title, status="waiting"):
    if status == "running":
        st.info(f"⏳ **Step {number}: {title}** — Running...")
    elif status == "done":
        st.success(f"✅ **Step {number}: {title}** — Completed")
    else:
        st.write(f"○ **Step {number}: {title}**")


def section_header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()


# ============================================================
# 1. SIMPLE CHAIN
# ============================================================

simple_prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

User:
{input}

Give a clear and helpful response.
""",
    input_variables=["input"]
)

simple_chain = simple_prompt | groq_llm | parser


def run_simple_chain(user_input):

    st.subheader("🔵 Simple Chain")

    st.caption(
        "The basic LangChain flow: Prompt → LLM → Output Parser"
    )

    st.markdown("### 🔄 Chain Flow")

    st.code(
        "📥 Input  →  📝 Prompt  →  🧠 LLM  →  ⚙️ Parser  →  📤 Output",
        language="text"
    )

    st.divider()

    show_step(1, "Input received", "done")

    show_step(2, "Prompt creation", "running")
    time.sleep(0.3)

    formatted_prompt = simple_prompt.invoke(
        {"input": user_input}
    )

    show_step(2, "Prompt creation", "done")

    show_step(3, "LLM generation", "running")

    response = groq_llm.invoke(formatted_prompt)

    show_step(3, "LLM generation", "done")

    show_step(4, "Output parsing", "running")

    final_output = parser.invoke(response)

    show_step(4, "Output parsing", "done")

    st.divider()

    st.subheader("📤 Final Output")

    st.success(final_output)

    return final_output


# ============================================================
# 2. SEQUENTIAL CHAIN
# ============================================================

prompt1 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a key point provider. Give important key points."
    ),
    (
        "human",
        "{user_input}"
    )
])

prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content writer. Write useful content based on the key points."
    ),
    (
        "human",
        "{key}"
    )
])


def run_sequential_chain(user_input):

    st.subheader("🟢 Sequential Chain")

    st.caption(
        "The output of the first chain becomes the input of the second chain."
    )

    st.markdown("### 🔄 Chain Flow")

    st.code(
        "📥 Input → 📝 Prompt 1 → 🧠 LLM → ⚙️ Parser "
        "→ 📝 Prompt 2 → 🧠 LLM → ⚙️ Parser → 📤 Output",
        language="text"
    )

    st.divider()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    show_step(1, "Input received", "done")

    show_step(2, "Generating key points", "running")

    step1_prompt = prompt1.invoke(
        {"user_input": user_input}
    )

    step1_response = gemini_llm.invoke(step1_prompt)

    key_points = parser.invoke(step1_response)

    show_step(2, "Generating key points", "done")

    with st.expander("🔑 Intermediate Output — Key Points", expanded=True):
        st.write(key_points)

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    show_step(3, "Writing content from key points", "running")

    step2_prompt = prompt2.invoke(
        {"key": key_points}
    )

    step2_response = gemini_llm.invoke(step2_prompt)

    final_output = parser.invoke(step2_response)

    show_step(3, "Writing content from key points", "done")

    st.divider()

    st.subheader("📤 Final Output")

    st.success(final_output)

    return final_output


# ============================================================
# 3. PARALLEL CHAIN
# ============================================================

docs = """
Large Language Models are AI systems capable of understanding
and generating human language.

LLMs are trained on large amounts of text data.

They can perform tasks such as:
- Text generation
- Summarization
- Question answering
- Translation
- Classification
- Code generation

Modern LLM applications often use prompt engineering,
retrieval augmented generation, tools and agents.
"""

prompt_notes = PromptTemplate(
    template="""
Create short and useful study notes from the following text.

TEXT:
{text}
""",
    input_variables=["text"]
)

prompt_quiz = PromptTemplate(
    template="""
Create 5 quiz questions from the following text.

TEXT:
{text}
""",
    input_variables=["text"]
)

prompt_merge = PromptTemplate(
    template="""
You are given study notes and quiz questions.

Study Notes:
{notes}

Quiz:
{quiz}

Combine them into a clean learning resource.
""",
    input_variables=["notes", "quiz"]
)

notes_chain = prompt_notes | groq_llm | parser

quiz_chain = prompt_quiz | openrouter_llm | parser

parallel_chain = RunnableParallel(
    {
        "notes": notes_chain,
        "quiz": quiz_chain
    }
)

merge_chain = prompt_merge | groq_llm | parser


def run_parallel_chain(user_text):

    st.subheader("🟣 Parallel Chain")

    st.caption(
        "Independent branches execute separately and their results are merged."
    )

    st.markdown("### 🔄 Chain Flow")

    st.code(
        """
                         ┌→ 📝 Notes → 🧠 LLM ─┐
📥 Input → RunnableParallel                  ├→ 🔀 Merge → 📤 Output
                         └→ ❓ Quiz  → 🧠 LLM ─┘
""",
        language="text"
    )

    st.divider()

    show_step(1, "Input received", "done")

    show_step(2, "Running parallel branches", "running")

    parallel_result = parallel_chain.invoke(
        {"text": user_text}
    )

    show_step(2, "Running parallel branches", "done")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Notes Branch")

        with st.container(border=True):
            st.write(parallel_result["notes"])

    with col2:
        st.subheader("❓ Quiz Branch")

        with st.container(border=True):
            st.write(parallel_result["quiz"])

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    show_step(3, "Merging branch outputs", "running")

    merged_prompt = prompt_merge.invoke(
        {
            "notes": parallel_result["notes"],
            "quiz": parallel_result["quiz"]
        }
    )

    merged_response = groq_llm.invoke(merged_prompt)

    final_output = parser.invoke(merged_response)

    show_step(3, "Merging branch outputs", "done")

    st.divider()

    st.subheader("📤 Final Merged Output")

    st.success(final_output)

    return final_output


# ============================================================
# 4. CONDITIONAL CHAIN
# ============================================================

class GetFeedback(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


feedback_parser = PydanticOutputParser(
    pydantic_object=GetFeedback
)

prompt_feedback = PromptTemplate(
    template="""
Classify the sentiment of the following feedback
into positive or negative.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction":
            feedback_parser.get_format_instructions()
    }
)

prompt_positive = PromptTemplate(
    template="""
Provide an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

prompt_negative = PromptTemplate(
    template="""
Provide an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)


positive_chain = prompt_positive | groq_llm | parser

negative_chain = prompt_negative | groq_llm | parser

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        positive_chain
    ),
    (
        lambda x: x.sentiment == "negative",
        negative_chain
    ),
    RunnableLambda(
        lambda x: "Could not find any sentiment."
    )
)


def run_conditional_chain(feedback):

    st.subheader("🟠 Conditional Chain")

    st.caption(
        "The chain first classifies the input and then selects a branch."
    )

    st.markdown("### 🔄 Chain Flow")

    st.code(
        """
📥 Feedback
     ↓
🧠 Sentiment Classifier
     ↓
   ┌───────────────┐
   ↓               ↓
😊 Positive      😞 Negative
   ↓               ↓
📝 Response      📝 Response
   └───────┬───────┘
           ↓
       📤 Output
""",
        language="text"
    )

    st.divider()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    show_step(1, "Feedback received", "done")

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    show_step(2, "Classifying sentiment", "running")

    classification_prompt = prompt_feedback.invoke(
        {"feedback": feedback}
    )

    classification_response = groq_llm.invoke(
        classification_prompt
    )

    sentiment_result = feedback_parser.invoke(
        classification_response
    )

    show_step(2, "Classifying sentiment", "done")

    if sentiment_result.sentiment == "positive":

        st.success("😊 Positive sentiment detected")

        selected_branch = "Positive Response"

        selected_chain = positive_chain

    else:

        st.warning("😞 Negative sentiment detected")

        selected_branch = "Negative Response"

        selected_chain = negative_chain

    st.info(f"🔀 Selected Branch: **{selected_branch}**")

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    show_step(
        3,
        f"Executing {selected_branch}",
        "running"
    )

    final_output = selected_chain.invoke(
        {"feedback": feedback}
    )

    show_step(
        3,
        f"Executing {selected_branch}",
        "done"
    )

    st.divider()

    st.subheader("📤 Final Output")

    st.success(final_output)

    return final_output


# ============================================================
# HEADER
# ============================================================

st.title("🧠 LangChain Chain Lab")

st.caption(
    "Interactive playground for understanding how different "
    "LangChain chains execute."
)

st.info(
    "💡 Select a chain from the sidebar and watch its execution step-by-step."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧩 Chain Lab")

st.sidebar.caption("Choose a LangChain pattern")

selected_chain = st.sidebar.radio(
    "Select Chain",
    [
        "🔵 Simple Chain",
        "🟢 Sequential Chain",
        "🟣 Parallel Chain",
        "🟠 Conditional Chain"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📚 Available Patterns")

st.sidebar.write("🔵 Simple")
st.sidebar.write("🟢 Sequential")
st.sidebar.write("🟣 Parallel")
st.sidebar.write("🟠 Conditional")

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python + LangChain + Streamlit"
)


# ============================================================
# SIMPLE UI
# ============================================================

if selected_chain == "🔵 Simple Chain":

    section_header(
        "🔵 Simple Chain",
        "One input flows through Prompt → LLM → Parser."
    )

    user_input = st.text_area(
        "Enter your input",
        placeholder="Ask something...",
        height=150
    )

    if st.button(
        "▶ Run Simple Chain",
        type="primary",
        use_container_width=True
    ):

        if not user_input.strip():

            st.warning("Please enter something first.")

        else:

            run_simple_chain(user_input)


# ============================================================
# SEQUENTIAL UI
# ============================================================

elif selected_chain == "🟢 Sequential Chain":

    section_header(
        "🟢 Sequential Chain",
        "One chain's output becomes the next chain's input."
    )

    user_input = st.text_input(
        "Enter a topic",
        placeholder="Example: Cricket, AI, Python..."
    )

    if st.button(
        "▶ Run Sequential Chain",
        type="primary",
        use_container_width=True
    ):

        if not user_input.strip():

            st.warning("Please enter a topic.")

        else:

            run_sequential_chain(user_input)


# ============================================================
# PARALLEL UI
# ============================================================

elif selected_chain == "🟣 Parallel Chain":

    section_header(
        "🟣 Parallel Chain",
        "Multiple independent chains process the same input."
    )

    user_text = st.text_area(
        "Enter document / topic",
        value=docs,
        height=250
    )

    if st.button(
        "▶ Run Parallel Chain",
        type="primary",
        use_container_width=True
    ):

        if not user_text.strip():

            st.warning("Please enter some text.")

        else:

            run_parallel_chain(user_text)


# ============================================================
# CONDITIONAL UI
# ============================================================

elif selected_chain == "🟠 Conditional Chain":

    section_header(
        "🟠 Conditional Chain",
        "The next step depends on the detected sentiment."
    )

    feedback = st.text_area(
        "Enter customer feedback",
        placeholder="Example: The application is amazing and very easy to use!",
        height=150
    )

    if st.button(
        "▶ Run Conditional Chain",
        type="primary",
        use_container_width=True
    ):

        if not feedback.strip():

            st.warning("Please enter feedback.")

        else:

            run_conditional_chain(feedback)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 LangChain Chain Lab • Simple • Sequential • Parallel • Conditional"
)