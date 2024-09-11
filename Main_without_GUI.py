
# !pip -q install langchain tiktoken chromadb pypdf transformers InstructorEmbedding sentence-transformers==2.2.2
# !pip -q install accelerate bitsandbytes

# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# # make sure that langchain is installed
# !pip show langchain 
# ! pip install yt_dlp
# ! pip install pydub

from langchain.llms import HuggingFacePipeline


from Models import *
from RAG_class import *

# Load the pipeline
HF_gen_pipeline = HF_Pipeline()
local_llm = HuggingFacePipeline(pipeline = HF_gen_pipeline.pipe)

# check if the pipeline is working, uncomment the next line to test
# print(local_llm('What is the capital of England?'))

video_URL = "https://www.youtube.com/watch?v=7Pq-S557XQU"

rag = RAG()


rag.process_YT_url(video_URL,local_llm )

# Make a query
query = "what will happen to human jobs?"

_ = rag.make_query(query)
