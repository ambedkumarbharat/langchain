from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Provide 5 positive and negative key on text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Explain all key with point wise on text \n {text}",
    input_variables=['text']
)

llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free"
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)

feedback = input("Enter your key : ")

result = chain.invoke({"text":feedback})

print(result)

chain.get_graph().print_ascii()