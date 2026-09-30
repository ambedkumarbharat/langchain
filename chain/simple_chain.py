from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        you are mentor and best assistant also provide best facts
"""),
("human","{human}")
])

llm = ChatGroq(model="openai/gpt-oss-120b", temperature= 0.7)

parser = StrOutputParser()

chain = prompt | llm | parser




print("-----------------wellcome to ai(simple chain) chatbot-------------------")
print("Enter 0,quit,exit to exit the chat ")

while True:
    user_input = input("You: ")
    if user_input == 0 or user_input == "quit" or user_input == "exit":
        break
    else:
        print("Ai: ", chain.invoke({"human" : user_input}))


# For a chain graph:
# chain.get_graph().print_ascii()
