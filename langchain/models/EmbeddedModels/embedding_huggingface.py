from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = 'model_name')

text = "random"

result = embedding_model.embed_query(text)

print(str(result))

# for more texts
documents = [
    "this is sentence 1",
    "this is sentence 2"
]

result = embedding_model.embed_documents(documents)

print(str(result))


