import os  
import sys  
import time  
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
import torch  
from threading import Lock  
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger  
from config.evaluator_config import EvaluatorConfig
from utils.logger import setup_logger



class EvaluatorDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



class BaseEvaluator:
    def __init__(self, config: EvaluatorConfig):
        """
        初始化评估器
        """
        self.config = config
        self.logger = setup_logger(name = __class__.__name__, level = "INFO")

        self.test_data = self.load_test_dataset(self.config.test_dataset_path)

        self.test_dataloader = DataLoader(
            EvaluatorDataset(self.test_data),
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
        )



    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载测试数据集
        """
        data = None
        if dataset_path.endswith(".json"):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析错误: {e}")
        else:
            raise ValueError(f"我们当前只支持 json， 不支持其他的数据集格式: {dataset_path}")

        return data



    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        评估数据集
        """
        raise NotImplementedError("子类必须实现evaluate_one_sample方法")
    


    def evaluate_batch_examples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:

        raise NotImplementedError("子类必须实现evaluate_batch_examples方法")

    
    def evaluate(self) -> Dict[str, float]:
        """
        评估数据集
        """
        raise NotImplementedError("子类必须实现evaluate方法")


    