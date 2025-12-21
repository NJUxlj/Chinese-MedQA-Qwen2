import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()


class KGConfig(BaseModel):
    """知识图谱配置"""
    uri: str = Field(default=os.getenv("NEO4J_URI"))
    username: str = Field(default=os.getenv("NEO4J_USERNAME"))
    password: str = Field(default=os.getenv("NEO4J_PASSWORD"))
