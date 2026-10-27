from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


model = ChatOpenAI(model = 'gpt-4' , temperature=0.3 , max_completion_tokens=50)
# max_completion_tokens , will determine how many words will be in the response , more tokens == will need more money , so it is important.
# temperature ranges between 0 to 2 , higher the value more creative will be the chatmodel be.



result = model.invoke("What is the capital of india?")

print(result)

