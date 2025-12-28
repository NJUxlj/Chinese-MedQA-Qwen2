import os, sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__)).parent.parent)
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any
from config.base_config import BaseConfig

from dotenv import load_dotenv

load_dotenv()


class LDAConfig(BaseModel):
    embedding_model_name: str = Field(...)
    language: str = Field(default = "chinese")
    offline_mode: bool = Field(default = False)
    
