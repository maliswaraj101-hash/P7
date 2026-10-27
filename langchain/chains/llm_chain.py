# rather than writing a prompt Template taking input and then giving it to model and printing the output.
# langchainc creaters build a LLMChain class that takes in the llm and prompt and does the rest of the work.
# one more layer of abstraction got added.


from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field
from langchain.chains import LLMChain
from typing import Literal




load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

prompt = PromptTemplate(
    template = "Suggest a catchy blog with title about {topic}",
    input_variables = ['topic']
)

chain = LLMChain(llm = llm , prompt = prompt)


topic = input("Enter a topic")

output = chain.run(topic)

print(output)

