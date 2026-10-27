from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFacePipeline
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",  
     huggingfacehub_api_token="YOUR_HF_TOKEN",
     task='text-generation'
)

# we are using hugging face api to run these models

# to download model and run them locally rather than api do this

# llm2 = HuggingFacePipeline.from_model_id(
#     model_id='model_name',
#     task = 'text-generation',
#     pipeline_kwargs=dict(
#         temperature = 0.5,
#         max_new_tokens = 100
#     )
# )

model = ChatHuggingFace(llm = llm)

resilt = model.invoke("What is the capital of delhi?")

print(resilt.content)