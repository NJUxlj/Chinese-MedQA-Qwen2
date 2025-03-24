
from ftllm import llm

from model.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from transformers import AutoTokenizer

from config.config import MODEL_PATH

import fastllm



import os
FLM_MODEL_PATH =  os.path.join( os.path.dirname(MODEL_PATH), "qwen2_medical.flm")

class FastLLMModel:
    def __init__(self, model:Qwen2ForCausalLM, tokenizer:AutoTokenizer, flm_model_path = None, device=None):
        
        if flm_model_path is not None:
            self.model = llm.model(flm_model_path)
            return
        
        self.model_name_or_path = model.config.name_or_path
        self.model = llm.from_hf(model, tokenizer, dtype="float16")
        
        
        
    
    
    def save_flm(self):
        self.model.save(FLM_MODEL_PATH)
        
        
