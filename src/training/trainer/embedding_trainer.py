import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)
from transformers import BitsAndBytesConfig

from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.logger import setup_logger
from config.training_config import EmbeddingTrainingConfig
from training.trainer.base_trainer import BaseTrainer



class EmbeddingTrainer(BaseTrainer):
    def __init__(self, config: EmbeddingTrainingConfig):
        super().__init__(config)

