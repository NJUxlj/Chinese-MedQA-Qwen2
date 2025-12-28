import os, sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__)).parent.parent)
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any
from config.base_config import BaseConfig

from dotenv import load_dotenv

load_dotenv()


class MilvusConfig(BaseConfig):
    """Milvus 配置"""
    uri: str = Field(default="http://localhost:19530")
    token: str = Field(default="root:Milvus")
    db_name: str = Field(default="default")
    index_type: str = Field(default="FLAT")
    metric_type: str = Field(default="L2")
    consistency_level: str = Field(default="Strong")
