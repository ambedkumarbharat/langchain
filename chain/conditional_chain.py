from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
class get_feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="give the sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object=get_feedback)

prompt_feedback = PromptTemplate(
    template = "Classify the sentiment of following text into positive or negative n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt_positive = PromptTemplate(
    template="provide an appropiate response to this positive feedback n {feedback}",
    input_variables=['feedback']
)

prompt_negative = PromptTemplate(
    template="provide an appropiate response to this negative feedback n {feedback}",
    input_variables=['feedback']
)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature= 0.7)

parser = StrOutputParser()

classify_chain = prompt_feedback | llm | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_positive | llm | parser),
    (lambda x: x.sentiment == 'negative', prompt_negative | llm | parser),
    RunnableLambda(lambda x: "Couldnot find any sentiment")
)

chain = classify_chain | branch_chain

feedback = input("Enter your feedback : ")

result = chain.invoke({"feedback":feedback})

print(result)

chain.get_graph().print_ascii()