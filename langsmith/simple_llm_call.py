
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# alternate way to set project name
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM chain'


# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key="YOUR_GROQ_API_KEY"
    )

parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | parser

config = {
    'tages' : {'llm app' , 'demo'},
    'metadata' : {'parser' : 'StrOutputParser'},
    
}
# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)