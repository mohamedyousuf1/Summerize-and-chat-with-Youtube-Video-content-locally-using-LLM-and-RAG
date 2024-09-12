from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

class HF_Pipeline:
    def __init__(self):
        # load tokenizer from google/flan-t5-xl
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        # load model from google/flan-t5-xl
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",
                                                    load_in_8bit=True,
                                                    device_map='auto',
                                                    cache_dir = 'Transformers_cache',
                                                    #   torch_dtype=torch.float16,
                                                    #   low_cpu_mem_usage=True,

                                                    )

        # initilize the pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # load summarizer from facebook/bart-large-cnn
        self.summarizer  = pipeline("summarization", model="facebook/bart-large-cnn")

    # summerize the list of texts that are extracted and splitted already
    def summerize_list_texts(self,texts):
        summerized_text = ""
        for text in texts:
            if len(text.page_content) > 100:
                summ_text = self.summarizer(text.page_content, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
                summ_text += '\n'
                summerized_text += summ_text
        
        return summerized_text
