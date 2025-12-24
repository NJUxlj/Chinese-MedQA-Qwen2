import logging
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional, Any, Callable, Union, Dict
from pydantic import BaseModel, Field, validator
from config.base_config import BaseConfig


class VLLMDeployerConfig(BaseConfig):
    model_name_or_path: str = Field(default = "Qwen/Qwen3-14B", description="模型名称或路径")
