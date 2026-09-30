from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt_tweet = PromptTemplate(
    template="You are content Writter create tweet content ont Topic.\n {text} ",
    input_variables=["text"]
)

prompt_linkedin = PromptTemplate(
    template="You are content Writter create LinkedIn Post on Topic.\n {text} ",
    input_variables=["text"]
)

llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free"
)

parser = StrOutputParser()

chain = RunnableParallel({
    "tweet":RunnableSequence(prompt_tweet, llm, parser),
    'linkedIn':RunnableSequence(prompt_linkedin, llm, parser)
})

input_text = input("Enter a topic: ")

result = chain.invoke({'text':input_text})

print("Tweet: ",result['tweet'])
print("LinkedIn: ",result['linkedIn'])

chain.get_graph().print_ascii()