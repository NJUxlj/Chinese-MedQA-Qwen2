from dataclasses import dataclass  
from typing import Optional, Dict , List, Tuple, Callable
import os  
import torch  
from torch.utils.data import Dataset  
from transformers import (  
    TrainingArguments,  
    AutoTokenizer,  
    # Qwen2ForCausalLM,  
    BitsAndBytesConfig,
    Trainer,  
    TrainerCallback  
)  

from model.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model  
from datasets import load_dataset  
# from trl import DPOTrainer  
# import deepspeed  
from deepspeed import DeepSpeedEngine 


class CustomDPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "chosen_labels": self.data["chosen_labels"][idx],  
            "rejected_labels": self.data["rejected_labels"][idx]  
        }  
        
        


class CustomDPOTrainer:  
    def __init__(  
        self,  
        output_dir: str,  
        dataset_name_or_path: str,  
        model_name: str = "qwen/qwen2-7b",  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = True,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        beta: float = 0.1  
    ):  
        self.output_dir = output_dir  
        self.dataset_name_or_path = dataset_name_or_path  
        self.beta = beta  
        self.max_seq_length = max_seq_length  

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset() 