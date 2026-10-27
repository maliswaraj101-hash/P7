from langchain_community.document_loaders import TextLoader

loader = TextLoader('info.txt' , encoding = 'utf-8')
# loader = PyPDFLoader('info.txt')
# for loading PDFS
# in pypdfloader , docs[0] is page 1 text , then docs[1] is page 2 , it scans page by page

docs = loader.load()

# print(docs)
print(docs[0])
# print(type(docs))

# all document loaders convert the info to list type