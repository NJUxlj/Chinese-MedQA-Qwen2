import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()


class RerankerConfig(BaseModel):
    """ reranker 配置
    
    用于从本地加载 Qwen3-reranker 这样的模型进行重排序
    """
    model_provider: str = Field(default="sentence_transformers", description="模型提供方：sentence_transformers, transformers")
    model_name: str = Field(default="Qwen3-Reranker-0.6B", description="模型名称")
    model_path: str = Field(default="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Reranker-0.6B", description="本地模型路径")
    device: str = Field(default="cpu", description="设备：cpu, cuda")
    batch_size: int = Field(default=32, description="批处理大小")
    normalize_scores: bool = Field(default=True, description="是否对分数进行归一化")
