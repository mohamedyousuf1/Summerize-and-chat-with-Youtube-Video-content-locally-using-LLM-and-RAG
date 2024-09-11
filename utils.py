import textwrap
import re
import requests
from PIL import Image
from io import BytesIO

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return llm_response

def get_youtube_thumbnail_image(youtube_url):
    # Extract the video ID from the YouTube URL
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    
    if video_id_match:
        video_id = video_id_match.group(1)
        # Get the max resolution thumbnail URL
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        # Download the image from the URL
        response = requests.get(thumbnail_url)
        
        if response.status_code == 200:
            # Open the image using PIL
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print("Thumbnail not found or not available.")
            return None
    else:
        print("Invalid YouTube URL.")
        return None


