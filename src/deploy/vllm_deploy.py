import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
import torch  
from threading import Lock  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger  

from config.vllm_deployer_config import VllmDeployerConfig

try:  
    from vllm import LLM, SamplingParams  
except ImportError:  
    raise ImportError("vLLM未安装，无法使用vLLM进行推理。请使用以下命令安装vLLM：pip install vllm")  

import subprocess


class VllmDeployer:
    def __init__(self, config: VllmDeployerConfig) -> None:
        self.config = config
        self.model_name_or_path = config.model_name_or_path
        self.tensor_parallel_size = config.tensor_parallel_size
        self.pipeline_parallel_size = config.pipeline_parallel_size
        self.gpu_utilization = config.gpu_utilization
        self.device = config.device
        self.lock = Lock()

        self.logger = setup_logger(name = __class__.__name__, level = "INFO")

    

    def deploy(self):
        '''

        '''


    
