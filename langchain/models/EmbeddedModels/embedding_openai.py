from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

# this is for hugging face api not local.

embedding_model = OpenAIEmbeddings(model = 'model_name' , dimensions=32)
# 32 dimension vector [5 7] , this is a 2 dimension vector , which will be in 2D , therefore higher dimension means more info will be captured , but more cost tooo via tha apis


result = embedding_model.embed_query("I am swaraj") 

print(str(result))


# for more text

documents = [
    "this is sentence 1",
    "this is sentence 2"
]

result = embedding_model.embed_documents(documents)

print(str(result))

