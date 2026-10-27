from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = 'field1' , description = 'this is field 1'),
    ResponseSchema(name = 'field2' , description = 'this is field 2'),
    ResponseSchema(name = 'field3' , description = 'this is field 3'),
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'give 3 facts about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

# with chains

chain = template | model | parser
result = chain.invoke({'topic' : 'black hole'})

print(result)


# structured output parsers enforces a json schema but doesnt do data validation , like here age can be a string too , if we want age to be a int , we need data validation , that can be done by pydantic output parser.

