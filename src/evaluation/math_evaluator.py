import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logger import setup_logger
from utils.metrics import DPOMetrics
from config.evaluator_config import DPOQualityEvaluatorConfig, MathEvaluatorConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset




class MathEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: MathEvaluatorConfig
    ):
        super().__init__(config)
        self.config = config



    def evaluate_one_sample(self):
        pass





    def evaluate_batch_samples(self, batch: Dict[str, Any]):
        pass




    def evaluate(self):
        pass








def run():
    pass







if __name__ == '__main__':
    run()
