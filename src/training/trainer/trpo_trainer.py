import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger
from config.training_config import TRPOTrainingConfig


class TRPOTrainer:
    """
    TRPO (Trust-Region Policy Optimization) Trainer
    
    支持：
    1. 全量微调 (Full Fine-tuning)
    2. LoRA 微调 (Low-Rank Adaptation)
    3. 4-bit/8-bit 量化
    4. DeepSpeed 加速
    5. 多卡分布式训练 (Accelerate)
    6. LoRA adapter 保存和合并
    """
    
    def __init__(
        self,
        config: TRPOTrainingConfig,
        finetuning_type: str = "lora",
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Dict] = None
    ):
        """
        初始化 TRPO 训练器
        
        Args:
            config: 训练配置
            finetuning_type: 微调类型 ("lora" 或 "full")
            use_deepspeed: 是否使用 DeepSpeed
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
        """
        self.config = config
        self.finetuning_type = finetuning_type
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        self.lora_config = None
        
        self.logger = setup_logger(self.__class__.__name__, level="INFO")
        
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError(f"finetuning_type 必须是 'lora' 或 'full'，但得到了 '{self.finetuning_type}'")
        
        if self.use_deepspeed and self.config.use_4bit:
            self.logger.warning("DeepSpeed 与 4-bit 量化可能存在兼容性问题，请谨慎使用")