from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFacePipeline
from huggingface_hub import InferenceClient


from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()


llm = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",  
      huggingfacehub_api_token="YOUR_HF_TOKEN",
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)





st.header('Research tool')

country_input = st.selectbox("Select country " , ["India" , "USA" , "China"])
city_input = st.selectbox("Select city " , ["Amsterdam" , "Mumbai" , "Beijing"])


template = load_prompt('template.json') 

prompt = template.invoke({
    'city_input' : city_input,
    'country_input' : country_input
})


if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)



# we create this kind of template to give the user less control , as we know models are highly sensitive even in a slight change of input prompt , thus we should not give the user much control of the prompt , therefore such prompt templates are created.
# these are called dynamic prompts
# in such application which we have created above , the model doesnt have the past history of chats , so if i say my name is swaraj and i move on to give a new prompt and ask what is my name , model wont know its swaraj
# we need to maintain a chat history , whenever we prompt and whatever the model gives the output , we need to save it in chat history.
# in the chat history , we should also let the model know which sentence was sent by the user and which sentence was sent by himself (model).
# for this we use messages.

from langchain_core.messages import SystemMessage , HumanMessage , AIMessage

messages = [
    SystemMessage(content = 'You are a AI assistant'),
    HumanMessage(content = 'Tell me about langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(result.content))


# these messages are the chat history , this is the way we store it , there will only be one systemMEssage and else humanmessage and aimessage will continue.
# this is a chatbot example

chat_history = [
    SystemMessage('You are a AI assistant')
]

while True:
    user_input = input("You :")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))


# this is a chatbot
 