from langchain_core.prompts import ChatPromptTemplate ,  MessagesPlaceholder
from langchain_core.messages import SystemMessage , HumanMessage

chat_template = ChatPromptTemplate([
    ('system' , 'You are a helpgul {domain} expert'),
    ('human' , 'Explain in simple terms , what is {topic}')
])


# chat_template = ChatPromptTemplate([
#     SystemMessage(content =  'You are a helpgul {domain} expert'),
#     HumanMessage(content = 'Explain in simple terms , what is {topic}')
# ])
# both of the above syntax is fine


prompt = chat_template.invoke({'domain':'cricket' , 'topic' : 'googly'})

print(prompt)

# but we need a chat history for model to undestand previous convos too right , for that we use message placeholders.



chat_template = ChatPromptTemplate([
    ('system' , 'You are a helpful customer support agent expert'),
    MessagesPlaceholder(variable_name = 'chat_history')
    ('human' , '{query}')
])

chat_history = []

prompt = chat_template.invoke({'chat_history' : chat_history , 'query' : 'where is my refund?'})

# prompt willl look like
# 1 --> system message  
# 2 --> chat_history
# 3 --> human message which will be the query



# PromptTemplate = fill-in-the-blank string → for raw LLMs.

# ChatPromptTemplate = fill-in-the-blank messages → for chat-based LLMs.

# Both do the same templating, but ChatPromptTemplate keeps the message-role structure, which matches how chat models are trained → more reliable control.


