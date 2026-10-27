from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser ,  ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class Summary(BaseModel):
    summary : str = Field(..., description="Summary of the text")
    no_of_words : int = Field(..., description="Number of words in the summary")
    sentiment : Literal['positive' , 'negative'] = Field(description = "Give the sentiment of the summary")

parser2 = PydanticOutputParser(pydantic_object = Summary)


llm = ChatGroq(
     api_key = "YOUR_GROQ_API_KEY",
     model = "openai/gpt-oss-120b"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic} ',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate summary for this {text}\n' \
    'Return ONLY a valid JSON object that matches the following format:\n'\
    '{format_instruction}\n\n',
    input_variables = ['text'],
    partial_variables = {'format_instruction' : parser2.get_format_instructions()}
)


schema = [
    ResponseSchema(name = 'field1' , description = 'this is field 1'),
    ResponseSchema(name = 'field2' , description = 'this is field 2'),
    ResponseSchema(name = 'field3' , description = 'this is field 3'),
]

parser3 = StructuredOutputParser.from_response_schemas(schema)

# and then the usual parser3.get_format_instructions() can be used to get the format instructions for the structured output parser.



chain = prompt | llm | parser | prompt2 | llm | parser2

result = chain.invoke({'topic' : 'cricket'})

print(result)

# print(prompt.format(topic s= 'Basketball'))