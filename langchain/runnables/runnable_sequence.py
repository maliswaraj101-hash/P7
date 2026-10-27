from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)





prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt , model , parser)



result = chain.invoke({'topic' : 'AI'})

print(result)