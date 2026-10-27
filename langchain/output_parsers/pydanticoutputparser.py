from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser , ResponseSchema
from pydantic import BaseModel , Field


load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description = 'name of the person')
    age : int = Field(gt = 18 , description = 'age of the person')
    city : str = Field(description = "name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = 'generate the name , age and city of a fictional {place} person \n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

print(parser.get_format_instructions())

prompt = template.invoke({'place' : 'indian'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

# this can also be done by chains

chain = template | model | parser



final_result2 = chain.invoke({'place' : 'sri lankan'})
print(final_result2)




# https://chatgpt.com/share/68cddfff-7cd0-8006-9452-505f6046852d



# Got it 👍 Let me break it down step by step in simple words.

# 1. What are partial variables in LangChain?
# Normally, when you create a PromptTemplate, you define placeholders like {topic} or {name} that the user will fill later.

# But sometimes, you also want to fill some placeholders in advance (not by the user each time).
# That’s where partial variables come in → they are pre-filled values for the template.

# 👉 Example:

# from langchain.prompts import PromptTemplate

# template = "Give 3 facts about {topic} \n {format_instructions}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["topic"],  # user will fill this
#     partial_variables={"format_instructions": "Answer in bullet points"}  # pre-filled
# )
# Here:

# {topic} → user provides

# {format_instructions} → already filled once and for all

# So when you use it:

# prompt.format(topic="India")
# You’ll get:

# Give 3 facts about India
# Answer in bullet points
# 2. What is get_format_instructions()?
# This is usually used with output parsers in LangChain.

# An output parser tells the LLM how the answer should look (e.g., JSON, dictionary, list, etc.).

# .get_format_instructions() gives a string that describes those rules.

# 👉 Example:

# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# # Define schema
# schemas = [
#     ResponseSchema(name="fact1", description="First fact"),
#     ResponseSchema(name="fact2", description="Second fact"),
#     ResponseSchema(name="fact3", description="Third fact"),
# ]

# parser = StructuredOutputParser.from_response_schemas(schemas)

# # Get format instructions
# print(parser.get_format_instructions())
# This might print something like:

# The output should be a JSON object with the following keys:
# - fact1: First fact
# - fact2: Second fact
# - fact3: Third fact
# 3. Putting it together
# Now, instead of hardcoding "Answer in bullet points", you use the parser’s format instructions:

# template = "Give 3 facts about {topic} \n {format_instructions}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["topic"],
#     partial_variables={"format_instructions": parser.get_format_instructions()}
# )
# So whenever you format it with a topic, it automatically adds the parser’s instructions.

# ✅ In simple words:

# Partial variables = pre-filled placeholders in your prompt.

# get_format_instructions() = tells the LLM the exact structure it must follow (e.g., JSON schema).

# Together → you don’t need to rewrite format rules every time, you just plug them in once.

# Do you want me to also show you a real working example with GPT + parser where it outputs JSON automatically?

# You said:
# parital variable are pre filled by the developers or coders , and users fill the input variables 
# ChatGPT said:
# Exactly ✅ You got it.

# Input variables → left empty for the user to provide each time.

# Partial variables → pre-filled once by the developer/coder (so the user doesn’t have to give them).

# 👉 Think of it like a form:

# Input variables = blank fields that the user must fill (like "Name" or "Topic").

# Partial variables = fixed text already written by the developer (like "Instructions: Write in JSON").

# Would you like me to make a small analogy example (like filling a Google Form) to make it super clear?

# You said:
# like get_format_instructions , what all functionality is available for such variables can u give more examples
# ChatGPT said:
# Nice question 👍
# You’re asking: apart from get_format_instructions() → what else can we plug into partial variables?

# Let’s break it down:

# 🔹 What can go inside partial variables?
# Partial variables are just pre-filled strings (or functions returning strings).
# So you can put:

# Static text (fixed instructions)

# Dynamic values (e.g., current date/time, user profile info)

# Functions that return strings (e.g., get_format_instructions() from parsers, or your own function)

# 🔹 Examples
# 1. Static pre-fill (simple text)
# from langchain.prompts import PromptTemplate

# template = "Write a {poem_type} about {topic}.\n{style}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["poem_type", "topic"],   # user fills these
#     partial_variables={"style": "Make it rhyme and use simple English."}
# )

# print(prompt.format(poem_type="haiku", topic="moon"))
# ➡ Output:

# Write a haiku about moon.
# Make it rhyme and use simple English.
# 2. Dynamic pre-fill (like today’s date)
# import datetime
# from langchain.prompts import PromptTemplate

# def get_date():
#     return f"Today's date is {datetime.date.today()}."

# template = "Summarize the news about {topic}.\n{date_info}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["topic"],
#     partial_variables={"date_info": get_date()}
# )

# print(prompt.format(topic="AI in healthcare"))
# ➡ Output (if today is 2025-09-18):

# Summarize the news about AI in healthcare.
# Today's date is 2025-09-18.
# 3. Using get_format_instructions()
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# schemas = [
#     ResponseSchema(name="title", description="The headline of the article"),
#     ResponseSchema(name="summary", description="Short summary"),
# ]

# parser = StructuredOutputParser.from_response_schemas(schemas)

# template = "Summarize the following article:\n{article}\n{format_instructions}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["article"],
#     partial_variables={"format_instructions": parser.get_format_instructions()}
# )

# print(prompt.format(article="AI is changing the world of medicine."))
# ➡ Output (prompt sent to LLM):

# Summarize the following article:
# AI is changing the world of medicine.

# The output should be a JSON object with the following keys:
# - title: The headline of the article
# - summary: Short summary
# 4. Custom reusable rules
# Imagine you always want polite answers:

# template = "Answer the question: {question}\n{tone}"

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["question"],
#     partial_variables={"tone": "Be polite, concise, and respectful."}
# )

# print(prompt.format(question="What is LangChain?"))
# ✅ So in short:

# partial_variables can hold any string (fixed, dynamic, or function output).

# Most common use cases:

# Reusable style/tone rules

# Formatting rules (like get_format_instructions())

# Context info (date, location, role, etc.)

# Would you like me to also show you an example where partial variable itself is a function that executes at runtime (not pre-computed once)? That’s a slightly advanced but powerful use case.

