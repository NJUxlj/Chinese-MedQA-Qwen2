from pathlib import Path
import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class NERConfig(BaseModel):
    """命名实体识别配置"""
    model_provider: str = Field(default="spacy")  # spacy or transformers
    model_name: str = Field(default="zh_core_web_sm")
    device: int = Field(default=-1)  # -1 for CPU, 0 for GPU
    use_auth_token: Optional[str] = Field(default=None)
    aggregation_strategy: str = Field(default="simple")  # simple, first, average, max
    batch_size: int = Field(default=32)
