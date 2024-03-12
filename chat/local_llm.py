import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.llms import HuggingFacePipeline

class LocalLLM:
    
    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.2'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 4 bit quantization
        _bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=_bnb_config,
            low_cpu_mem_usage=True
        )
        
        _text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        
        self.llm = HuggingFacePipeline(pipeline=_text_generation_pipeline)
        


            