import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()

from config.base_config import BaseConfig



class InferenceConfig(BaseConfig):
    # 推理配置 
    model_name_or_path: str = Field(default="Qwen/Qwen3-14B", description="微调后的模型路径")


class VLLMConfig(BaseConfig):
    """ VLLM 配置
    """
    # 推理配置 
    model_name_or_path: str = Field(default="Qwen/Qwen3-14B", description="微调后的模型路径")
    serving_name: str = Field(default="Qwen3-14B", description="服务名称")
    max_new_tokens: int = Field(default=1024, description="最大新令牌数")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.9, description="Top-p 采样参数")
    use_flash_attn: bool = Field(default=False, description="是否使用 Flash Attention")
    tensor_parallel_size: int = Field(default=1, description="张量并行大小")
    pipeline_parallel_size: int = Field(default=1, description="管道并行大小")

    # 服务配置  
    host: str = Field(default="0.0.0.0", description="服务主机地址 (default: 0.0.0.0)")
    port: int = Field(default=8000, description="服务端口号 (default: 8000)")
    debug: bool = Field(default=False, description="是否开启调试模式 (default: False)")
    verbose: bool = Field(default=False, description="是否开启详细日志 (default: False)")




class TransformersGenerationConfig(InferenceConfig):
    pass