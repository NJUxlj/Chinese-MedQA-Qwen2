import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()


class RerankerConfig(BaseModel):
    """ reranker 配置"""
    model_provider: str = Field(default="transformers", description="模型提供方：transformers, sentence_transformers")
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="模型名称")
    device: str = Field(default="cpu", description="设备：cpu, cuda")
