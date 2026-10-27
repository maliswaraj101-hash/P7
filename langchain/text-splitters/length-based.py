from langchain_text_splitters import CharacterTextSplitter, Language , RecursiveCharacterTextSplitter 


text = "hello this is swaraj"

splitter = CharacterTextSplitter(
    chunk_size = 5,
    chunk_overlap = 0,
    separator = ''
)


# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 5,
#     chunk_overlap = 0,
# )

# for language like python 


splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 5,
    language= Language.PYTHON,
    chunk_overlap = 0,
)

# semantic meaning based splitting , 
# what this does is , it splits the text sentence wise and compares embeddings of each pair of sentences.
# if cosine similiarity is too low for 2 sentences means the topic about which they are speaking is not the same , so we make a split at that place.


result = splitter.split_text(text)


print(result)