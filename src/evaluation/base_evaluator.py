import json
import os
import sys
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import torch
from threading import Lock
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger


class EvaluatorDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BaseEvaluator:
    """所有评估器的基类。配置必须由调用方显式传入，不再读取全局 settings。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(name=self.__class__.__name__, level="INFO")


    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        if not dataset_path or dataset_path == "None":
            return []
        if not dataset_path.endswith(".json"):
            raise ValueError(f"目前只支持 .json 格式的数据集: {dataset_path}")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析错误: {e}")

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError("子类必须实现 evaluate_one_sample 方法")

    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        raise NotImplementedError("子类必须实现 evaluate_batch_samples 方法")

    def evaluate(self) -> Dict[str, float]:
        raise NotImplementedError("子类必须实现 evaluate 方法")
