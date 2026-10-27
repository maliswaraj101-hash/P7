from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Define the folder containing PDFs
pdf_folder_path = "path/to/your/pdf/folder"

# used to load a folder having many files
# Use DirectoryLoader with PyPDFLoader
loader = DirectoryLoader(
    pdf_folder_path,
    glob="*.pdf",                 # Load only PDF files
    loader_cls=PyPDFLoader        # Tell it to use PyPDFLoader for each file
)

# Load all PDFs in the directory
documents = loader.load()
#  instead of loader.load() , u can also use lazy_load()\
# what normal load does is if there are 500 pdfs , it will load all of them and print
# what lazy load does is , it fetches one at a time , not tries to do it  all at once

# Print out some information
print(f"Loaded {len(documents)} documents")
print(documents[0].page_content[:500])  # Preview first 500 chars of first PDF
