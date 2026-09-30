from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the ChatOpenAI client with your custom base URL and model
llm = ChatOpenAI(
    base_url="https://api.xkiro.com/v1",
    model="qwen/qwen3.8-omni-flash:free"
)

# Invoke the model with a message
response = llm.invoke("Hello!")

# Print the response content
print(response.content)