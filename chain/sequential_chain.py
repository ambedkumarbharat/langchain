from langchain_groq import ChatGroq
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = ChatPromptTemplate.from_messages([
    ("system","""
        you are just key point provider
"""),
("human","{user_input}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system"," you are content wrtitter on key "),
    ("human","{key}")
])



llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature= 0.7)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({"user_input": "cricket"})

print(result)

chain.get_graph().print_ascii()