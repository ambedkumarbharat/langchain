from dotenv import load_dotenv
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda

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

def token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))

    
sequential_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'key' : RunnablePassthrough(),
    'content': RunnableSequence(prompt2, llm, parser),
    'word_count': RunnableLambda(token_count)
})



chain = RunnableSequence(sequential_chain, parallel_chain)

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