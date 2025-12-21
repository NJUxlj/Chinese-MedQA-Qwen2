import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig(BaseModel):
    """知识图谱配置"""
    model_name: str = Field(default=os.getenv("EMBEDDING_MODEL_NAME"))