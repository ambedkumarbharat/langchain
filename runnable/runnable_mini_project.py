from dotenv import load_dotenv
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template="""
You are an expert learning-prompt generator.

The user will give you a simple request about something they want to learn.

Your task is to convert the user's request into a clear, detailed,
and structured prompt that another AI tutor can use to teach the user.

The generated prompt should:
- Understand the user's actual learning goal like road map.
- Identify the subject or topic.
- Assume beginner level unless the user specifies otherwise.
- Define what concepts should be covered.
- Arrange the learning topics logically from basic to advanced.
- Include practical examples and exercises where appropriate.
- Focus on understanding rather than memorization.
- Do not answer the user's question yourself.
- Output ONLY the final learning prompt.

User request:
{user_input}
""",
    input_variables=["user_input"]
)

prompt2 = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert learning-content organizer.

The user will provide a structured learning instruction.

Your task is to identify the most important concepts that the learner
needs to understand from that instruction.

Generate clear and concise key points.

Rules:
- Focus only on the important concepts.
- Arrange them in a logical learning order.
- Do not explain the concepts in detail.
- Do not ask questions.
- Do not add unrelated topics.
- Keep the output beginner-friendly.

Output only the key points.
"""),
    ("human", "{learning_prompt}")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI tutor.

Teach the learner using the provided learning instruction and key points.

Rules:
- Explain concepts in simple language.
- Follow the key points in order.
- Start from basic concepts and gradually increase difficulty.
- Give practical examples where useful.
- Focus on understanding, not memorization.
- Do not skip important concepts.
- Do not ask the learner questions unless specifically requested.
"""),
    ("human", """

Key Points:
{key_points}

Now explain the topic to the learner.
""")
])

llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free"
)

parser = StrOutputParser()

def token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))

    
sequential_chain_1 = RunnableSequence(prompt1, llm, parser)

sequential_chain_2 = RunnableSequence(prompt2, llm, parser)

parallel_chain = RunnableParallel({
    'key' : RunnablePassthrough(),
    'content': RunnableSequence(prompt3, llm, parser),
    'word_count': RunnableLambda(token_count)
})



chain = RunnableSequence(sequential_chain_1, sequential_chain_2, parallel_chain)

user_data = input('write a topic: ')

result = chain.invoke({'user_input': user_data})

print("Key: ",result['key'])

print()
print()

print("Explaination: ",result['content'])

print()
print()

print("Token Count: ",result['word_count'])


chain.get_graph().print_ascii()