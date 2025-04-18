
import os
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from ..dataset.medical_dataset import MedicalDataset
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SFTTrainer:
    """基于Hugging Face Trainer的SFT训练器"""
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        training_args: Optional[Dict[str, Any]] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.training_args = training_args or {}
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"Loading model and tokenizer from {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 确保分词器有正确的填充token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 为适应医疗问答对话，调整模型配置
        if hasattr(self.model.config, "max_length"):
            self.model.config.max_length = 2048
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, medical_dataset: MedicalDataset) -> Dataset:
        """准备训练数据集"""
        logger.info("Preparing dataset for SFT")
        return medical_dataset.get_sft_dataset(self.tokenizer)
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        """创建Trainer实例"""
        default_args = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "logging_steps": 10,
            "save_steps": 200,
            "evaluation_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 200 if eval_dataset else None,
            "save_total_limit": 3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "fp16": torch.cuda.is_available(),
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "load_best_model_at_end": True if eval_dataset else False,
        }
        
        # 更新默认参数
        for key, value in self.training_args.items():
            default_args[key] = value
        
        training_args = TrainingArguments(**default_args)
        
        def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """自定义数据整理函数"""
            batch = {}
            for key in features[0].keys():
                if key in ["input_ids", "attention_mask", "labels"]:
                    batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            return batch
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self, medical_dataset: MedicalDataset, eval_split: float = 0.1):
        """执行SFT训练"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # 准备数据集
        dataset = medical_dataset.get_sft_dataset(self.tokenizer)
        
        # 划分训练和评估数据集
        if eval_split > 0:
            dataset = dataset.train_test_split(test_size=eval_split)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # 创建训练器
        self.create_trainer(train_dataset, eval_dataset)
        
        # 开始训练
        logger.info("Starting SFT training")
        self.trainer.train()
        
        # 保存最终模型
        logger.info(f"Saving final model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return self.output_dir






if __name__ == "__main__":
    pass