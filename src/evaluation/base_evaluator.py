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
from config.settings import settings


class EvaluatorDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BaseEvaluator:
    """所有评估器的基类，从 settings.evaluator 读取统一配置。

    子类可在 __init__ 中通过 config_override 传入任意 OmegaConf / dict-like
    对象来覆盖默认的 settings.evaluator 配置。
    """

    def __init__(self, config=None):
        """
        Args:
            config: 可选，评估器配置对象（omegaconf DictConfig 或任意支持
                    属性访问的对象）。若为 None，则使用 settings.evaluator。
        """
        self.config = config if config is not None else settings.evaluator
        self.logger = setup_logger(name=self.__class__.__name__, level="INFO")

        dataset_path = getattr(self.config, "test_dataset_path", None)
        if dataset_path is not None:
            self.test_data = self.load_test_dataset(str(dataset_path))
        else:
            self.test_data = []

        batch_size = int(getattr(self.config, "batch_size", 4))
        num_workers = int(getattr(self.config, "dataloader_num_workers", 0))
        self.test_dataloader = DataLoader(
            EvaluatorDataset(self.test_data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = str(self.config.model_name_or_path)
        device = str(getattr(self.config, "device", "cpu"))
        padding_side = str(getattr(self.config, "padding_side", "left"))
        use_fast = bool(getattr(self.config, "use_fast", True))

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            use_fast=use_fast,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """加载测试数据集（目前仅支持 JSON 格式）"""
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

    def evaluate_batch_examples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        raise NotImplementedError("子类必须实现 evaluate_batch_examples 方法")

    def evaluate(self) -> Dict[str, float]:
        raise NotImplementedError("子类必须实现 evaluate 方法")
