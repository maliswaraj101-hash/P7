from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch , RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()


report_generator_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x  : len(x.split()) > 300 , prompt2 | model | parser),
    # this is condition
    RunnablePassthrough()
    # this is default

    # syntax is
    # condition()
    # condition()
    # default

    # (lambda x  : len(x.split()) > 300 , prompt2 | model | parser),
    # if x length is greater than 300 then the prompt2 | model | parser chain is run , if its not then the bydefault is run which is passthrough as we mentioned
)

chain = report_generator_chain | branch_chain

result = chain.invoke({'topic' : 'AI'})

print(result)