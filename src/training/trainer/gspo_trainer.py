from typing import List, Optional, Union
from pathlib import Path
import sys, os
sys.path.append(str(Path(__file__).parent.parent.parent))
from trainer.base_trainer import BaseTrainer
from config.settings import settings
from utils.logger import setup_logger


'''
GSPO 的训练样本中有 3 个字段： id, messages, ground_true_answer
- messages 中的前 n-1 轮， 就是传统意义上的 prompt
'''


class GSPOTrainer(BaseTrainer):
    def __init__(self, config: GSPOTrainingConfig):
        super().__init__(config)






    def prepare_dataset(self):
        pass




    def tokenize_dataset(self):
        pass




    def start_training(self):
        pass