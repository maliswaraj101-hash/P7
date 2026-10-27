
from langgraph.graph import START , END , StateGraph
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , AIMessage , HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.checkpoint.sqlite import SqliteSaver
from groq import Groq
import sqlite3
from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]
    
def chatnode(state : ChatState) -> ChatState:

    messages = state['messages']
   
    groq_messages = []
    for msg in messages:
        if isinstance(msg , HumanMessage):
            groq_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            groq_messages.append({"role": "assistant", "content": msg.content})
    client = Groq(api_key="YOUR_GROQ_API_KEY")

    completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=groq_messages,
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""

    return {"messages": [AIMessage(content=answer)]}


# checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chatnode" , chatnode)
graph.add_edge(START , 'chatnode')
graph.add_edge('chatnode' , END)

conn = sqlite3.connect('chatbot.db' , check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)
workflow = graph.compile(checkpointer = checkpointer)

# for streaming , instead of workflow.invoke we use workflow.stream

# for message_chunk , metdata in workflow.stream(
#     {'messages' : [HumanMessage(content = 'write a blog on rohit sharma ')]},
#     config = {'configurable' : {'thread_id' : "1"}},
#     stream_mode='messages'
# ):
#     if(message_chunk.content):
#         print(message_chunk.content , end= " " , flush = True)





# response = workflow.invoke(
#             {"messages": [HumanMessage(content="what is my name")]},
#             config={"configurable": {"thread_id": "thread-1"}},
#         )

# print(response)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
