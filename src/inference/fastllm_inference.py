import os  
import sys  
import ctypes  
from typing import Dict, List, Optional, Union, Any, Tuple  
import numpy as np  
from models.base_model import BaseModel  
from utils.logger import setup_logger  

logger = setup_logger(__name__)  

class FastLLMInference(BaseModel):  
    """使用FastLLM进行加速推理"""  
    
    def __init__(  
        self,  
        model_path: str,  
        device: str = "cpu",  # "cpu", "cuda", "cuda:0", etc.  
        max_length: int = 2048,  
        lib_path: Optional[str] = None,  
    ):  
        self.model_path = model_path  
        self.device = device  
        self.max_length = max_length  
        self.fastllm_loaded = False  
        
        # 尝试加载FastLLM库  
        self.fastllm = None  
        try:  
            if lib_path:  
                self.fastllm = ctypes.CDLL(lib_path)  
            else:  
                # 尝试从默认位置加载  
                system = sys.platform  
                if system == "linux":  
                    self.fastllm = ctypes.CDLL("libfastllm.so")  
                elif system == "win32":  
                    self.fastllm = ctypes.CDLL("fastllm.dll")  
                elif system == "darwin":  
                    self.fastllm = ctypes.CDLL("libfastllm.dylib")  
                else:  
                    raise ValueError(f"Unsupported system: {system}")  
            
            self.fastllm_loaded = True  
            logger.info("FastLLM library loaded successfully")  
            
            # 加载模型  
            self.model = self._load_model()  
            
        except Exception as e:  
            logger.error(f"Failed to load FastLLM: {e}")  
            logger.warning("Falling back to regular model inference")  
            
            # 这里可以添加回退逻辑，例如使用Hugging Face的模型  
            self.fastllm_loaded = False  
    
    def _load_model(self):  
        """加载FastLLM模型"""  
        if not self.fastllm_loaded:  
            return None  
        
        # 设置FastLLM库的函数参数类型和返回类型  
        self.fastllm.LoadModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p]  
        self.fastllm.LoadModel.restype = ctypes.c_void_p  
        
        self.fastllm.Generate.argtypes = [  
            ctypes.c_void_p,   
            ctypes.c_char_p,   
            ctypes.c_int,   
            ctypes.c_float,   
            ctypes.c_float  
        ]  
        self.fastllm.Generate.restype = ctypes.c_char_p  
        
        self.fastllm.DeleteModel.argtypes = [ctypes.c_void_p]  
        
        # 设置设备  
        device_param = self.device.encode('utf-8')  
        
        # 加载模型  
        model_ptr = self.fastllm.LoadModel(  
            self.model_path.encode('utf-8'),  
            device_param  
        )  
        
        if not model_ptr:  
            raise RuntimeError(f"Failed to load model from {self.model_path}")  
        
        logger.info(f"Model loaded successfully from {self.model_path}")  
        return model_ptr  
    
    def generate(  
        self,   
        prompt: str,   
        max_new_tokens: Optional[int] = None,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
    ) -> str:  
        """生成文本"""  
        if not self.fastllm_loaded or not self.model:  
            raise RuntimeError("FastLLM not loaded or model not initialized")  
        
        max_tokens = max_new_tokens or self.max_length  
        
        # 使用FastLLM进行生成  
        response = self.fastllm.Generate(  
            self.model,  
            prompt.encode('utf-8'),  
            max_tokens,  
            temperature,  
            top_p  
        )  
        
        # 解码并返回响应  
        return response.decode('utf-8')  
    
    def __del__(self):  
        """析构函数，释放模型资源"""  
        if self.fastllm_loaded and self.model:  
            try:  
                self.fastllm.DeleteModel(self.model)  
                logger.info("FastLLM model resources released")  
            except:  
                pass  


# inference/vllm_inference.py  

import os  
from typing import Dict, List, Optional, Union, Any, Tuple  
from vllm import LLM, SamplingParams  
from models.base_model import BaseModel  
from utils.logger import setup_logger  

logger = setup_logger(__name__)  

class VLLMInference(BaseModel):  
    """使用VLLM进行加速推理"""  
    
    def __init__(  
        self,  
        model_path: str,  
        tensor_parallel_size: int = 1,  
        gpu_memory_utilization: float = 0.9,  
        max_model_len: Optional[int] = None,  
        quantization: Optional[str] = None,  # "int8", "int4", None  
        trust_remote_code: bool = True,  
    ):  
        self.model_path = model_path  
        self.tensor_parallel_size = tensor_parallel_size  
        self.gpu_memory_utilization = gpu_memory_utilization  
        self.max_model_len = max_model_len  
        self.quantization = quantization  
        self.trust_remote_code = trust_remote_code  
        
        # 初始化VLLM模型  
        self.model = self._load_model()  
    
    def _load_model(self):  
        """加载VLLM模型"""  
        logger.info(f"Loading model from {self.model_path} with VLLM")  
        
        # 设置量化选项  
        dtype = "auto"  
        if self.quantization == "int8":  
            dtype = "int8"  
        elif self.quantization == "int4":  
            dtype = "int4"  
        
        try:  
            model = LLM(  
                model=self.model_path,  
                tensor_parallel_size=self.tensor_parallel_size,  
                gpu_memory_utilization=self.gpu_memory_utilization,  
                max_model_len=self.max_model_len,  
                dtype=dtype,  
                trust_remote_code=self.trust_remote_code,  
            )  
            
            logger.info("VLLM model loaded successfully")  
            return model  
            
        except Exception as e:  
            logger.error(f"Failed to load VLLM model: {e}")  
            raise RuntimeError(f"Failed to load VLLM model: {e}")  
    
    def generate(  
        self,   
        prompt: str,   
        max_new_tokens: Optional[int] = None,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        top_k: Optional[int] = None,  
        stop_tokens: Optional[List[str]] = None,  
    ) -> str:  
        """使用VLLM生成文本"""  
        if not self.model:  
            raise RuntimeError("VLLM model not initialized")  
        
        # 构建采样参数  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            top_k=top_k,  
            max_tokens=max_new_tokens or 1024,  
            stop=stop_tokens,  
        )  
        
        # 执行生成  
        outputs = self.model.generate(prompt, sampling_params)  
        
        # 提取生成的文本  
        generated_text = outputs[0].outputs[0].text  
        
        return generated_text  
    
    def batch_generate(  
        self,  
        prompts: List[str],  
        max_new_tokens: Optional[int] = None,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        top_k: Optional[int] = None,  
        stop_tokens: Optional[List[str]] = None,  
    ) -> List[str]:  
        """批量生成文本"""  
        if not self.model:  
            raise RuntimeError("VLLM model not initialized")  
        
        # 构建采样参数  
        sampling_params = SamplingParams(  
            temperature=temperature,  
            top_p=top_p,  
            top_k=top_k,  
            max_tokens=max_new_tokens or 1024,  
            stop=stop_tokens,  
        )  
        
        # 执行批量生成  
        outputs = self.model.generate(prompts, sampling_params)  
        
        # 提取生成的文本  
        generated_texts = [output.outputs[0].text for output in outputs]  
        
        return generated_texts  