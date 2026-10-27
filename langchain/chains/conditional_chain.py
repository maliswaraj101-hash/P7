from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field
from typing import Literal


# NOW WE WILL MAKE A APPLICATION WITH THIS FLOW
#  Input: {"text": "this is a terrible code"}
#              │
#              ▼
#  ┌──────────────────────────┐
#  │  Prompt1: Sentiment Task │
#  │  "Classify text as       │
#  │   positive/negative"     │
#  └──────────────────────────┘
#              │
#              ▼
#  ┌──────────────────────────┐
#  │ HuggingFace Model (Gemma)│
#  │   + Pydantic Parser      │
#  └──────────────────────────┘
#              │
#              ▼
#      Sentiment Object
#      ┌──────────────────┐
#      │ Feedback(         │
#      │   sentiment=...   │
#      │ )                 │
#      └──────────────────┘
#              │
#              ▼
#  ┌─────────────────────────────┐
#  │ Branch (RunnableBranch)     │
#  │ if sentiment == "positive"  │
#  │    → Prompt2 → Model → Resp │
#  │ if sentiment == "negative"  │
#  │    → Prompt3 → Model → Resp │
#  │ else                        │
#  │    → "could not find..."    │
#  └─────────────────────────────┘
#              │
#              ▼
#  ┌──────────────────────────┐
#  │ Final Output: Response   │
#  │ e.g.                     │
#  │ "I’m sorry you feel that │
#  │ way. Could you explain…" │
#  └──────────────────────────┘



load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):

    sentiment : Literal['positive' , 'negative'] = Field(description = "Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = ("Classify the sentiment of the following feedback text as either 'positive' or 'negative'.\n"
        "Return ONLY a valid JSON object that matches the following format:\n"
        "{format_instruction}\n\n"
        "Feedback: {text}"),
    input_variables = ['text'],
    partial_variables = {'format_instruction' : parser2.get_format_instructions()}
)

# we are using pydantic output parser cause we want only 2 words to be the output , either "positive" or "negative"




prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {text} ',
    input_variables = ['text'],
)



prompt3 = PromptTemplate(
     template = 'Write an appropriate response to this negative feedback \n {text} ',
    input_variables = ['text'],
)



chain = prompt1 | model | parser2


branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive' , prompt2 | model | parser),
    (lambda x : x.sentiment == 'negative' , prompt3 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment")

)

# this is like if else , if sentiment is positive then do this chain else if sentiment is negative then do this chain , else "could not find sentiment"
# we used RunnableLambda because it converts lambda function to a chain , RunnableBranch only accepts chain , therefore we converted lambda function to a chain using RunnableLambda

final_chain = chain | branch_chain

result = final_chain.invoke({"text" : "this is a terrible code"})

print(result)




# When you call

# result = final_chain.invoke({"text": "this is a terrible code"})
# the input you provide must match the input variables of the very first chain in your pipeline (prompt1).

# Why?
# Your pipeline is:

# final_chain = (prompt1 | model | parser2) | branch_chain
# The first runnable here is prompt1, which has:

# input_variables = ["text"]
# So the outermost chain (final_chain) also expects {"text": ...} as input.

# If you had defined prompt1 with variables like:

# PromptTemplate(input_variables=["feedback_text"])
# then you would need to invoke with:

# final_chain.invoke({"feedback_text": "this is a terrible code"})
# ✅ Rule of thumb:
# 👉 The input to .invoke() must match the input schema of the first runnable in your chain.

