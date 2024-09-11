
# !pip -q install langchain tiktoken chromadb pypdf transformers InstructorEmbedding sentence-transformers==2.2.2
# !pip -q install accelerate bitsandbytes gradio

# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# # make sure that langchain is installed
# !pip show langchain 
# ! pip install yt_dlp
# ! pip install pydub

from langchain.llms import HuggingFacePipeline
import gradio as gr
import numpy as np
from Models import *
from RAG_class import *

class process_video:
    def __init__(self):
        # Load the pipeline
        self.video_url = None
        self.HF_gen_pipeline = HF_Pipeline()
        self.local_llm = HuggingFacePipeline(pipeline = self.HF_gen_pipeline.pipe)
        # check if the pipeline is working, uncomment the next line to test
        # print(local_llm('What is the capital of England?'))
        self.rag = RAG()


    def process_url_button(self, video_url):
        self.video_url = video_url
        self.rag.process_YT_url( self.video_url, self.local_llm)
        return get_youtube_thumbnail_image(self.video_url)
    
    def process_query_button(self, query):
        return self.rag.make_query_GUI(query)



# Make a query

if __name__ == "__main__":

    prcess_video_class = process_video()
    with gr.Blocks() as demo:
        # gr.Label("Enter a query to search for in the video:")
        with gr.Row():
            with gr.Column():
                video_url = gr.Text( label="Enter the video URL", placeholder="https://www.youtube.com/watch?v=7Pq-S557XQU")
                URLbutton = gr.Button("Process Youtube URL")
                query = gr.Textbox( label="Enter your query here", placeholder="What is the capital of England?")
                query_button = gr.Button("Generate Answer")
            Thumbnail_image = gr.Image(label="Youtube Thumbnail Image")

        with gr.Row():

            result = gr.Text()

        URLbutton.click(fn=prcess_video_class.process_url_button , inputs=[video_url], outputs=[Thumbnail_image])
        query_button.click(fn= prcess_video_class.process_query_button , inputs=[query], outputs=[result])
    demo.launch(share=True)
