
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
        self.summerize_list_texts = None


    def process_url_button(self, video_url):
        self.video_url = video_url
        self.summerize_list_texts = self.rag.process_YT_url( self.video_url, self.local_llm)
        return get_youtube_thumbnail_image(self.video_url)
    
    def process_summary_button(self):
        summerized_text = self.HF_gen_pipeline.summerize_list_texts(self.summerize_list_texts)
        if len(summerized_text) > 4000:
            summerized_summerized_text = ''
            sum_texts = self.rag.text_splitter.split_text(summerized_text)
            for text in sum_texts:
                if len(text) > 100:
                    summ_text = self.HF_gen_pipeline.summarizer(text, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
                    summ_text += '\n'
                    summerized_summerized_text += summ_text  
            return summerized_summerized_text

        return summerized_text
        
        return summerized_text
    
    def process_query_button(self, query):
        return self.rag.make_query_GUI(query)


if __name__ == "__main__":

    prcess_video_class = process_video()
    with gr.Blocks() as demo:
        # gr.Label("Enter a query to search for in the video:")
        with gr.Row():
            with gr.Column():
                video_url = gr.Text( label="Enter the video URL", placeholder="https://www.youtube.com/watch?v=7Pq-S557XQU")
                with gr.Row():
                    URLbutton = gr.Button("Process Youtube URL")
                    summerize_button = gr.Button("Summerize Video Content")

                query = gr.Textbox( label="Enter your query here")
                query_button = gr.Button("Generate Answer")
            Thumbnail_image = gr.Image(label="Youtube Thumbnail Image")

        with gr.Row():
            summerize_output = gr.Textbox( label="Video Summary", placeholder="The video summerization will be displayed here")
            result = gr.Text()

        URLbutton.click(fn=prcess_video_class.process_url_button , inputs=[video_url], outputs=[Thumbnail_image])
        summerize_button.click(fn=prcess_video_class.process_summary_button , outputs=[summerize_output])
        query_button.click(fn= prcess_video_class.process_query_button , inputs=[query], outputs=[result])
    demo.launch(share=True)
