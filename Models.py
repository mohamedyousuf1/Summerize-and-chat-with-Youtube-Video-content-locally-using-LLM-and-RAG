from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

class HF_Pipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",
                                                    load_in_8bit=True,
                                                    device_map='auto',
                                                    cache_dir = 'Transformers_cache',
                                                    #   torch_dtype=torch.float16,
                                                    #   low_cpu_mem_usage=True,

                                                    )


        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        

