import logging
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional, Any, Callable, Union, Dict
from pydantic import BaseModel, Field, validator
from config.base_config import BaseConfig


class VLLMDeployerConfig(BaseConfig):
    model_name_or_path: str = Field(default="Qwen/Qwen3-14B", description="模型名称或路径")
    tensor_parallel_size: int = Field(default=1, description="张量并行大小，设置为GPU数量进行多卡推理")
    pipeline_parallel_size: int = Field(default=1, description="流水线并行大小，用于多节点推理")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.0, le=1.0, description="GPU显存利用率，范围0-1")
    max_model_len: Optional[int] = Field(default=None, description="最大模型上下文长度，None则自动推断")
    max_num_seqs: Optional[int] = Field(default=None, description="最大同时处理的序列数")
    dtype: str = Field(default="auto", description="数据类型: auto, half, float, bfloat16, half+float")
    quantization: Optional[str] = Field(default=None, description="量化方式: None, awq, gptq, squeezellm")
    enforce_eager: bool = Field(default=False, description="是否强制使用eager模式（禁用CUDA图）")
    swap_space: int = Field(default=4, description="CPU与GPU间交换空间大小(GB)")
    trust_remote_code: bool = Field(default=True, description="是否信任远程代码")
    seed: int = Field(default=42, description="随机种子")
    enable_auto_tool_choice: bool = Field(default=False, description="是否启用自动工具选择")
    tool_call_pattern: Optional[list] = Field(default=None, description="工具调用模式")
