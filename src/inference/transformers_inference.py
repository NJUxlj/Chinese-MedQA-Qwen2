import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
import torch  
from threading import Lock  
from pathlib import Path

# 确保可以导入项目其他模块  【也就是src目录下的其他包】
sys.path.append(str(Path(__file__)).parent.parent)  

from utils.logger import setup_logger  
from config.local_model_config import LocalModelConfig  
from inference.inference_utils import (  
    format_prompt,   
    format_rag_prompt,   
    postprocess_response,   
    measure_latency,  
    get_model_path,  
    get_inference_params  
)  



try:  
    from transformers import (
        AutoModelForCausalLM,  
        AutoTokenizer,  
        pipeline,  
        set_seed,  
        __version__ as transformers_version
    )
except ImportError:  
    raise ImportError("transformers 未安装，无法使用 transformers 进行推理。请使用以下命令安装：pip install transformers")  


class TransformersInference:  
    """  
    基于vLLM的推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, model_path: str, *args, **kwargs):  
        """  
        实现单例模式，相同路径的模型只加载一次  
        """  
        with cls._lock:     # 加锁保证线程安全
            if model_path not in cls._instances:    # # 检查是否已有实例
                instance = super(TransformersInference, cls).__new__(cls)    # 创建新实例 【调用父类的 __new__ 方法来创建 TransformersInference 类的一个新实例】
                cls._instances[model_path] = instance     # 存入字典
            return cls._instances[model_path]  # 返回单例
    
    @staticmethod
    def _get_torch_dtype(dtype: str):
        """将字符串dtype转换为torch dtype"""
        dtype_map = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype, None)

    def __init__(self,
                model_path: str,
                dtype: str = "auto",
                tensor_parallel_size: int = 1,
                gpu_memory_utilization: float = 0.8,
                max_model_len: int = 4096,
                quantization: Optional[str] = None):  
        """  
        初始化Transformers推理类  
        
        Args:  
            model_path: 模型路径  
            dtype: 数据类型，auto、float16、bfloat16或float32  
            tensor_parallel_size: 张量并行大小  
            gpu_memory_utilization: GPU显存使用率  
            max_model_len: 最大模型长度  
            quantization: 量化方法，awq、squeezellm或gptq  
        """  
        
        # 防止重复初始化  
        if hasattr(self, 'model'):  
            return  

        
        self.logger = setup_logger("vllm_inference", level="INFO")  
            
        self.model_path = get_model_path(model_path)  
        self.dtype = dtype  
        self.tensor_parallel_size = tensor_parallel_size  
        self.gpu_memory_utilization = gpu_memory_utilization  
        self.max_model_len = max_model_len  
        self.quantization = quantization  
        
        self.logger.info(f"正在加载模型: {self.model_path}")  
        start_time = time.time()  
        
        try:  
            # 配置GPU选项  
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0  
            if gpu_count < self.tensor_parallel_size:  
                self.logger.warning(f"可用GPU数量({gpu_count})小于请求的张量并行大小({self.tensor_parallel_size})，将使用所有可用GPU")  
                self.tensor_parallel_size = max(1, gpu_count)  
            
            # 加载模型 (使用 transformers 的 AutoModelForCausalLM，不支持 vLLM 特有参数)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self._get_torch_dtype(self.dtype),
                trust_remote_code=True
            )  
            
            # 获取分词器  
            self.tokenizer = self.model.get_tokenizer()  
            
            # 打印模型信息  
            self.logger.info(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")  
            
        except Exception as e:  
            self.logger.error(f"加载模型时出错: {str(e)}")  
            raise  
    
    @measure_latency   # 测量方法执行的延迟/耗时
    def generate(self,   
                prompt: str,   
                max_new_tokens: int = 512,  
                temperature: float = 0.7,  
                top_p: float = 0.9,  
                top_k: int = 20,  
                repetition_penalty: float = 1.1,  
                presence_penalty: float = 0.0,  
                frequency_penalty: float = 0.0,  
                do_sample: bool = True,  
                num_beams: int = 1,  
                stop: Optional[List[str]] = None,  
                **kwargs) -> str:  
        """  
        生成文本  
        
        Args:  
            prompt: 输入提示  
            max_new_tokens: 最大生成长度  
            temperature: 温度  
            top_p: top-p值  
            top_k: top-k值  
            repetition_penalty: 重复惩罚  
            presence_penalty: 存在惩罚  
            frequency_penalty: 频率惩罚  
            do_sample: 是否使用采样  
            num_beams: 光束搜索宽度  
            stop: 停止词列表  
            
        Returns:  
            生成的文本  
        """  