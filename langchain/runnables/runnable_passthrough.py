from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

# what passthrough does is if we give runnable passthrough a input X it will give the same input as the output 

# suppose we wanna make a application , model takes in a topic , outputs a joke about the topic , and then gives the explanation about the joke  

# the total output we get is only THE EXPLANATION , what if we want explanation + joke as an output 


# we will make a parallel runnable and a RunnableSequence



# prompt1 ---> 11m ---> parser --> RunnablePassthrough (outputs the joke)
                            #  --> prompt2 ---> 11m --> parser (outputs the explanation)


prompt1 = PromptTemplate(
    template = 'Tell me a joke about the topic - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the joke - {joke}',
    input_variables=['joke']
)


parser = StrOutputParser()

chain = RunnableSequence(prompt1 , model , parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2 , model , parser)
})

final_chain = RunnableSequence(chain , parallel_chain)

result = final_chain.invoke({'topic' : 'AI'})

print(result)


