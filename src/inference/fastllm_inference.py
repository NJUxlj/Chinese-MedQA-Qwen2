"""  
使用fastllm进行模型推理的模块  
fastllm是一个基于C++的推理库，能够在CPU和GPU上高效地运行LLM模型  
"""  
import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple  
import torch  
import numpy as np  
import ctypes  
from threading import Lock  

# 确保可以导入项目其他模块  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from utils.logger import get_logger  
from config.model_config import ModelConfig  
from inference.inference_utils import (  
    format_prompt,   
    format_rag_prompt,   
    postprocess_response,   
    measure_latency,  
    get_model_path  
)  

logger = get_logger("fastllm_inference")  

# 尝试导入fastllm  
try:  
    import fastllm  
    FASTLLM_AVAILABLE = True  
except ImportError:  
    logger.warning("fastllm未安装，无法使用fastllm进行推理。请按照以下步骤安装fastllm：")  
    logger.warning("1. git clone https://github.com/ztxz16/fastllm")  
    logger.warning("2. cd fastllm && mkdir build && cd build")  
    logger.warning("3. cmake .. -DUSE_CUDA=ON && make -j")  
    logger.warning("4. cd ../pybind && python setup.py install")  
    FASTLLM_AVAILABLE = False  

class FastLLMInference:  
    """  
    基于fastllm的推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, model_path: str, *args, **kwargs):  
        """  
        实现单例模式，相同路径的模型只加载一次  
        """  
        with cls._lock:  
            if model_path not in cls._instances:  
                instance = super(FastLLMInference, cls).__new__(cls)  
                cls._instances[model_path] = instance  
            return cls._instances[model_path]  
    
    def __init__(self, model_path: str, device: str = "cuda", load_8bit: bool = True):  
        """  
        初始化FastLLM推理类  
        
        Args:  
            model_path: 模型路径  
            device: 设备，cuda或cpu  
            load_8bit: 是否加载为8bit精度模型  
        """  
        if not FASTLLM_AVAILABLE:  
            raise ImportError("请先安装fastllm库")  
        
        # 防止重复初始化  
        if hasattr(self, 'model'):  
            return  
            
        self.model_path = get_model_path(model_path)  
        self.device = device  
        self.load_8bit = load_8bit  
        
        logger.info(f"正在加载模型: {self.model_path}")  
        start_time = time.time()  
        
        try:  
            # 加载模型  
            self.model = fastllm.model(self.model_path)  
            
            # 设置设备  
            if device == "cuda" and torch.cuda.is_available():  
                self.model.to_device("cuda")  
                logger.info("模型已加载到CUDA设备")  
            else:  
                logger.info("模型已加载到CPU设备")  
            
            # 如果需要转换为8bit  
            if load_8bit and device == "cuda":  
                self.model.to_device_int8("cuda")  
                logger.info("模型已转换为8bit精度")  
            
            # 尝试识别模型类型  
            self.model_type = self._detect_model_type()  
            
            # 创建对应的分词器  
            self.tokenizer = None  # fastllm内部处理分词  
            
            # 打印模型信息  
            logger.info(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")  
            
        except Exception as e:  
            logger.error(f"加载模型时出错: {str(e)}")  
            raise  
    
    def _detect_model_type(self) -> str:  
        """  
        检测模型类型  
        
        Returns:  
            模型类型字符串  
        """  
        model_name = os.path.basename(self.model_path).lower()  
        if "qwen" in model_name:  
            return "qwen"  
        elif "chatglm" in model_name:  
            return "chatglm"  
        elif "llama" in model_name or "baichuan" in model_name:  
            return "llama"  
        else:  
            return "unknown"  
    
    @measure_latency  
    def generate(self,   
                prompt: str,   
                max_new_tokens: int = 512,  
                temperature: float = 0.7,  
                top_p: float = 0.9,  
                repetition_penalty: float = 1.1,  
                **kwargs) -> str:  
        """  
        生成文本  
        
        Args:  
            prompt: 输入提示  
            max_new_tokens: 最大生成长度  
            temperature: 温度  
            top_p: top-p值  
            repetition_penalty: 重复惩罚  
            
        Returns:  
            生成的文本  
        """  
        try:  
            # 设置推理参数  
            generate_config = fastllm.GenerationConfig()  
            generate_config.max_length = max_new_tokens  
            generate_config.temperature = temperature  
            generate_config.top_p = top_p  
            generate_config.repeat_penalty = repetition_penalty  
            
            # 执行推理  
            response = self.model.chat(prompt, generate_config)  
            
            # 后处理  
            return postprocess_response(response)  
            
        except Exception as e:  
            logger.error(f"推理过程中出错: {str(e)}")  
            return f"推理出错: {str(e)}"  
    
    def answer_question(self,   
                        query: str,   
                        history: Optional[List[Dict[str, str]]] = None,  
                        system_prompt: Optional[str] = None,  
                        **kwargs) -> str:  
        """  
        回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Returns:  
            生成的回答  
        """  
        # 格式化提示词  
        prompt = format_prompt(query, history, system_prompt)  
        
        # 生成回答  
        return self.generate(prompt, **kwargs)  
    
    def answer_with_rag(self,  
                       query: str,  
                       context_docs: List[str],  
                       history: Optional[List[Dict[str, str]]] = None,  
                       system_prompt: Optional[str] = None,  
                       **kwargs) -> str:  
        """  
        使用RAG技术回答问题  
        
        Args:  
            query: 用户查询  
            context_docs: 从知识库检索到的相关文档  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Returns:  
            生成的回答  
        """  
        # 格式化RAG提示词  
        prompt = format_rag_prompt(query, context_docs, history, system_prompt)  
        
        # 生成回答  
        return self.generate(prompt, **kwargs)  
    
    def stream_generate(self,   
                       prompt: str,   
                       max_new_tokens: int = 512,  
                       temperature: float = 0.7,  
                       top_p: float = 0.9,  
                       repetition_penalty: float = 1.1,  
                       **kwargs):  
        """  
        流式生成文本  
        
        Args:  
            prompt: 输入提示  
            max_new_tokens: 最大生成长度  
            temperature: 温度  
            top_p: top-p值  
            repetition_penalty: 重复惩罚  
            
        Yields:  
            生成的文本片段  
        """  
        try:  
            # 设置推理参数  
            generate_config = fastllm.GenerationConfig()  
            generate_config.max_length = max_new_tokens  
            generate_config.temperature = temperature  
            generate_config.top_p = top_p  
            generate_config.repeat_penalty = repetition_penalty  
            
            # 执行推理  
            response_text = ""  
            for response in self.model.stream_chat(prompt, generate_config):  
                new_text = response[len(response_text):]  
                response_text = response  
                yield new_text  
                
        except Exception as e:  
            logger.error(f"流式推理过程中出错: {str(e)}")  
            yield f"推理出错: {str(e)}"  
    
    def unload(self):  
        """  
        卸载模型  
        """  
        if hasattr(self, 'model'):  
            del self.model  
            # 执行垃圾回收  
            import gc  
            gc.collect()  
            if torch.cuda.is_available():  
                torch.cuda.empty_cache()  
            
            # 从实例字典中移除  
            with self.__class__._lock:  
                if self.model_path in self.__class__._instances:  
                    del self.__class__._instances[self.model_path]  
                    
            logger.info(f"模型 {self.model_path} 已卸载")  


# 测试代码  
if __name__ == "__main__":  
    if not FASTLLM_AVAILABLE:  
        print("请先安装fastllm库")  
        sys.exit(1)  
        
    # 测试模型路径，请替换为实际的模型路径  
    model_path = "/path/to/your/model"  
    
    # 初始化推理类  
    inference = FastLLMInference(model_path)  
    
    # 测试普通问答  
    query = "高血压患者应该注意什么？"  
    print("问题:", query)  
    
    answer = inference.answer_question(query)  
    print("回答:", answer)  
    
    # 测试RAG问答  
    context_docs = [  
        "高血压患者应该控制盐的摄入量，每日不超过5克。",  
        "高血压患者应该适当增加体育锻炼，但避免剧烈运动。",  
        "高血压患者应该保持心情舒畅，避免精神紧张。"  
    ]  
    
    rag_answer = inference.answer_with_rag(query, context_docs)  
    print("RAG回答:", rag_answer)  
    
    # 测试流式生成  
    print("流式生成:")  
    for text in inference.stream_generate(format_prompt(query)):  
        print(text, end="", flush=True)  
    print()  