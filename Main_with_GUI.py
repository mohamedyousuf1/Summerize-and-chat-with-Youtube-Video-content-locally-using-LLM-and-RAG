
# !pip -q install langchain tiktoken chromadb pypdf transformers InstructorEmbedding sentence-transformers==2.2.2
# !pip -q install accelerate bitsandbytes gradio

# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# # make sure that langchain is installed
# !pip show langchain 
# ! pip install yt_dlp
# ! pip install pydub

from langchain.llms import HuggingFacePipeline
import gradio as gr
from Models import *
from RAG_class import *

class process_video:
    '''
    The process_video class contains the following methods:
        __init__: This method is used to initialize the process_video class.
        process_url_button: This method is used to process the youtube video URL.
        process_summary_button: This method is used to process the summary button.
        process_query_button: This method is used to process the query button.
        
        Attributes:
            video_url: The video URL.
            HF_gen_pipeline: The HF_gen_pipeline is loaded from the HF_Pipeline model.
            local_llm: The local_llm is loaded from the HuggingFacePipeline model.
            rag: The rag is loaded from the RAG model.
            summerize_list_texts: The summerize_list_texts is set to None.
            
        Methods:
            process_url_button: This method is used to process the youtube video URL.
            process_summary_button: This method is used to process the summary button.
            process_query_button: This method is used to process the query button.
            
    '''
    def __init__(self):
        # Load the pipeline
        self.video_url = None

        # Load the HF_gen_pipeline
        self.HF_gen_pipeline = HF_Pipeline()

        # Load the local_llm, which is used to generate the answer to the query
        self.local_llm = HuggingFacePipeline(pipeline = self.HF_gen_pipeline.pipe)
        # check if the pipeline is working, uncomment the next line to test
        # print(local_llm('What is the capital of England?'))

        # Load the rag, which is used to process the youtube video URL
        self.rag = RAG()
        self.summerize_list_texts = None


    def process_url_button(self, video_url):
        self.video_url = video_url

        # process the youtube video URL
        self.summerize_list_texts = self.rag.process_YT_url( self.video_url, self.local_llm)
        return get_youtube_thumbnail_image(self.video_url)
    
    def process_summary_button(self):
        summerized_text = self.HF_gen_pipeline.summerize_list_texts(self.summerize_list_texts)
        if len(summerized_text) > 4000:
            summerized_summerized_text = ''
            sum_texts = self.rag.text_splitter.split_text(summerized_text)
            for text in sum_texts:
                if len(text) > 100:
                    summ_text = self.HF_gen_pipeline.summarizer(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
                    summ_text += '\n'
                    summerized_summerized_text += summ_text  
            return summerized_summerized_text

        return summerized_text
            
    def process_query_button(self, query):
        return self.rag.make_query_GUI(query)


if __name__ == "__main__":

    prcess_video_class = process_video()
    with gr.Blocks() as demo:
        gr.Label("An experimental framework to summerize ad chat with the contentt of Youtube video")
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
