
import os
import json
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MedicalDataset:
    """医疗数据集处理类"""
    
    def __init__(
        self,
        data_path: str,
        dataset_type: str = "sft",  # "sft" 或 "dpo"
        max_length: int = 512,
        system_prompt: Optional[str] = None,
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.raw_data = []
        
        # 默认系统提示
        self.system_prompt = system_prompt or "你是一个专业的医疗助手，请根据你的知识回答以下医疗问题。"
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载医疗数据集"""
        logger.info(f"Loading medical dataset from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file {self.data_path} not found")
        
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)
        elif self.data_path.endswith(".jsonl"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        logger.info(f"Loaded {len(self.raw_data)} examples")
    
    def _format_sft_example(self, example: Dict) -> Dict:
        """格式化SFT样本"""
        query = example.get("query", "")
        response = example.get("response", "")
        
        if not query or not response:
            logger.warning("Empty query or response found in data")
            return None
        
        # 构建Qwen2的输入格式
        # 基于Chatml格式
        formatted_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        formatted_text += f"<|im_start|>user\n{query}<|im_end|>\n"
        formatted_text += f"<|im_start|>assistant\n{response}<|im_end|>"
        
        return {"text": formatted_text}
    
    def _format_dpo_example(self, example: Dict) -> Dict:
        """格式化DPO样本"""
        query = example.get("query", "")
        chosen = example.get("chosen", "") or example.get("response", "")
        rejected = example.get("rejected", "")
        
        if not query or not chosen or not rejected:
            logger.warning("Missing query, chosen or rejected response in DPO data")
            return None
        
        # 构建查询部分
        query_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        query_text += f"<|im_start|>user\n{query}<|im_end|>\n"
        query_text += f"<|im_start|>assistant\n"
        
        # 构建响应部分（只包含response，不包括之前的对话）
        chosen_text = chosen
        rejected_text = rejected
        
        return {
            "query": query_text,
            "chosen": chosen_text,
            "rejected": rejected_text
        }
    
    def get_sft_dataset(self, tokenizer: PreTrainedTokenizer) -> Dataset:
        """获取SFT格式的数据集"""
        if self.dataset_type != "sft":
            logger.warning(f"Dataset type is {self.dataset_type}, but requesting SFT dataset")
        
        # 格式化数据
        formatted_data = []
        for example in self.raw_data:
            formatted_example = self._format_sft_example(example)
            if formatted_example:
                formatted_data.append(formatted_example)
        
        dataset = Dataset.from_list(formatted_data)
        
        # 对数据进行分词处理
        def tokenize_function(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 为了训练语言模型，labels与input_ids相同
            outputs["labels"] = outputs["input_ids"].clone()
            
            return outputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def get_dpo_dataset(self, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None) -> Dataset:
        """获取DPO格式的数据集"""
        if self.dataset_type != "dpo":
            logger.warning(f"Dataset type is {self.dataset_type}, but requesting DPO dataset")
        
        max_length = max_length or self.max_length
        
        # 格式化数据
        formatted_data = []
        for example in self.raw_data:
            formatted_example = self._format_dpo_example(example)
            if formatted_example:
                formatted_data.append(formatted_example)
        
        dataset = Dataset.from_list(formatted_data)
        
        # 对数据进行分词处理
        def tokenize_function(examples):
            # 处理查询部分
            query_tokenized = tokenizer(
                examples["query"],
                truncation=True,
                max_length=max_length // 2,  # 分配一半长度给查询
                padding="max_length",
                return_tensors="np"
            )
            
            # 处理选择回复部分
            chosen_tokenized = tokenizer(
                examples["chosen"],
                truncation=True,
                max_length=max_length // 2,  # 分配一半长度给回复
                padding="max_length",
                return_tensors="np"
            )
            
            # 处理拒绝回复部分
            rejected_tokenized = tokenizer(
                examples["rejected"],
                truncation=True,
                max_length=max_length // 2,  # 分配一半长度给回复
                padding="max_length",
                return_tensors="np"
            )
            
            return {
                "query_ids": query_tokenized["input_ids"],
                "query_attention_mask": query_tokenized["attention_mask"],
                "chosen_ids": chosen_tokenized["input_ids"],
                "chosen_attention_mask": chosen_tokenized["attention_mask"],
                "rejected_ids": rejected_tokenized["input_ids"],
                "rejected_attention_mask": rejected_tokenized["attention_mask"],
            }
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["query", "chosen", "rejected"]
        )
        
        return tokenized_dataset

