from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = ChatPromptTemplate.from_messages([
    ("system","""
        you are just key point provider
"""),
("human","{user_input}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system"," you are content explainer wrtitter on key "),
    ("human","{key}")
])

llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free"
)

parser = StrOutputParser()


sequential_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'key': RunnablePassthrough(),
    'content': RunnableSequence(prompt2, llm, parser)
})

chain = RunnableSequence(sequential_chain, parallel_chain)

input_data = input("write a topic : ")

result = chain.invoke({"user_input": input_data})

print(result['key'])

print(result['content'])

chain.get_graph().print_ascii()