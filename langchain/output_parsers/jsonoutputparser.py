from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser  


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'give me the name , age and city of a fictional person {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

prompt = template.format()

print(prompt)


result = model.invoke(prompt)
print(parser.parse(result.content))

# you can also write in this way

chain = template | model | parser

result2 = chain.invoke({}) 

print(result2)

# issue with jsonoutputparser is , we cant enforce a schema , the LLM automatically decides the schema , if say we want
# 3 fields , name age email , we cant enforce the output to be in this schema via jsonoutputparser.