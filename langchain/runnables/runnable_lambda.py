from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    huggingfacehub_api_token="YOUR_HF_TOKEN",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

text_count = runnable_word_counter.invoke('this word has 5 counts')


parser = StrOutputParser()

prompt = PromptTemplate(
    template='tell me a joke about topic = {topic}',
    input_variables=['topic']
)


chain_for_joke = RunnableSequence(prompt | model | parser)

parallel_chainn = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
})

chain = RunnableSequence(chain_for_joke, parallel_chainn)


result = chain.invoke({'topic' : 'AI'})

print(result)




