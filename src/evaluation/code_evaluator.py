'''
评估模型在 SWE-bench, LiveCodeBench 等代码测试集上的性能
'''

import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
import torch  
from threading import Lock  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger  

from config.evaluator_config import CodeEvaluatorConfig

class CodeEvaluator:

    def __init__(self, config: CodeEvaluatorConfig):
        """
        初始化评估器
        """

        self.config = config


    def evaluate_one_sample(self, response, ground_true_answer):
        """
        """
