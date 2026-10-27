from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


# we will create 2 chains , one for creaitng linkeding post other for tweets

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables = ['topic']
)


prompt2 = PromptTemplate(
    template = 'Generate a Linkedin post about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1 , model , parser),
    'linkedin' : RunnableSequence(prompt2 , model , parser)
})



result = parallel_chain.invoke({'topic' : 'AI'})
print(result)
