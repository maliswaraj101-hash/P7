from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

# 1st prompt
# giving the LLM prompt to build a detailed report

template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ['topic']
)

# 2nd prompt will be giving the LLM prompt to make 5 line summary on above mentioned detail report
template2 = PromptTemplate(
    template = "write a 5 line summary on {text}",
    input_variables = ['text']
)



prompt1 = template1.invoke({'topic' : 'cricket'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})

result2 = model.invoke(prompt2)

print(result2.content)


# with string parser this can be done like this

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result3 = chain.invoke({'topic' : 'football'})
print(result3) 