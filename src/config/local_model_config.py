import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()



class LocalModelConfig(BaseModel):
    """ 本地模型配置
    """
    model_path: str = Field(default="/root/autodl-tmp/models/Qwen2.5-0.5B", description="本地模型路径")
    device: str = Field(default="cuda:0", description="设备")



