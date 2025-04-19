"""  
使用vLLM进行模型推理的模块  
vLLM是一个高性能的LLM推理引擎，支持高吞吐量和低延迟的文本生成  
"""  
import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
import torch  
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
    get_model_path,  
    get_inference_params  
)  

logger = get_logger("vllm_inference")  

# 尝试导入vLLM  
try:  
    from vllm import LLM, SamplingParams  
    VLLM_AVAILABLE = True  
except ImportError:  
    logger.warning("vLLM未安装，无法使用vLLM进行推理。请使用以下命令安装vLLM：")  
    logger.warning("pip install vllm")  
    VLLM_AVAILABLE = False  

class VLLMInference:  
    """  
    基于vLLM的推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, model_path: str, *args, **kwargs):  
        """  
        实现单例模式，相同路径的模型只加载一次  
        """  
        with cls._lock:  
            if model_path not in cls._instances:  
                instance = super(VLLMInference, cls).__new__(cls)  
                cls._instances[model_path] = instance  
            return cls._instances[model_path]  
    
    def __init__(self,   
                model_path: str,   
                dtype: str = "auto",   
                tensor_parallel_size: int = 1,  
                gpu_memory_utilization: float = 0.8,  
                max_model_len: int = 4096,  
                quantization: Optional[str] = None):  
        """  
        初始化vLLM推理类  
        
        Args:  
            model_path: 模型路径  
            dtype: 数据类型，auto、float16、bfloat16或float32  
            tensor_parallel_size: 张量并行大小  
            gpu_memory_utilization: GPU显存使用率  
            max_model_len: 最大模型长度  
            quantization: 量化方法，awq、squeezellm或gptq  
        """  
        if not VLLM_AVAILABLE:  
            raise ImportError("请先安装vLLM库")  
        
        # 防止重复初始化  
        if hasattr(self, 'model'):  
            return  
            
        self.model_path = get_model_path(model_path)  
        self.dtype = dtype  
        self.tensor_parallel_size = tensor_parallel_size  
        self.gpu_memory_utilization = gpu_memory_utilization  
        self.max_model_len = max_model_len  
        self.quantization = quantization  
        
        logger.info(f"正在加载模型: {self.model_path}")  
        start_time = time.time()  
        
        try:  
            # 配置GPU选项  
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0  
            if gpu_count < self.tensor_parallel_size:  
                logger.warning(f"可用GPU数量({gpu_count})小于请求的张量并行大小({self.tensor_parallel_size})，将使用所有可用GPU")  
                self.tensor_parallel_size = max(1, gpu_count)  
            
            # 加载模型  
            self.model = LLM(  
                model=self.model_path,  
                dtype=self.dtype,  
                tensor_parallel_size=self.tensor_parallel_size,  
                gpu_memory_utilization=self.gpu_memory_utilization,  
                max_model_len=self.max_model_len,  
                quantization=self.quantization,  
                trust_remote_code=True  
            )  
            
            # 获取分词器  
            self.tokenizer = self.model.get_tokenizer()  
            
            # 打印模型信息  
            logger.info(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")  
            
        except Exception as e:  
            logger.error(f"加载模型时出错: {str(e)}")  
            raise  
    
    @measure_latency  
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
        try:  
            # 设置采样参数  
            sampling_params = SamplingParams(  
                max_tokens=max_new_tokens,  
                temperature=0.0 if not do_sample else temperature,  
                top_p=1.0 if not do_sample else top_p,  
                top_k=-1 if not do_sample else top_k,  
                repetition_penalty=repetition_penalty,  
                presence_penalty=presence_penalty,  
                frequency_penalty=frequency_penalty,  
                stop=stop or ["<|im_end|>"],  
                n=1,  
                # best_of 相当于 num_beams  
                best_of=num_beams if not do_sample else 1  
            )  
            
            # 执行推理  
            outputs = self.model.generate(prompt, sampling_params)  
            response = outputs[0].outputs[0].text  
            
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
    
    def batch_generate(self,   
                      prompts: List[str],   
                      **kwargs) -> List[str]:  
        """  
        批量生成文本  
        
        Args:  
            prompts: 输入提示列表  
            
        Returns:  
            生成的文本列表  
        """  
        try:  
            # 设置采样参数  
            sampling_params = SamplingParams(  
                max_tokens=kwargs.get("max_new_tokens", 512),  
                temperature=0.0 if not kwargs.get("do_sample", True) else kwargs.get("temperature", 0.7),  
                top_p=1.0 if not kwargs.get("do_sample", True) else kwargs.get("top_p", 0.9),  
                top_k=-1 if not kwargs.get("do_sample", True) else kwargs.get("top_k", 20),  
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),  
                presence_penalty=kwargs.get("presence_penalty", 0.0),  
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),  
                stop=kwargs.get("stop", ["<|im_end|>"]),  
                n=1,  
                best_of=kwargs.get("num_beams", 1) if not kwargs.get("do_sample", True) else 1  
            )  
            
            # 执行批量推理  
            outputs = self.model.generate(prompts, sampling_params)  
            responses = [output.outputs[0].text for output in outputs]  
            
            # 后处理  
            return [postprocess_response(response) for response in responses]  
            
        except Exception as e:  
            logger.error(f"批量推理过程中出错: {str(e)}")  
            return [f"推理出错: {str(e)}"] * len(prompts)  
    
    def stream_generate(self,   
                       prompt: str,   
                       **kwargs) -> Iterator[str]:  
        """  
        流式生成文本  
        
        Args:  
            prompt: 输入提示  
            
        Yields:  
            生成的文本片段  
        """  
        try:  
            # 设置采样参数  
            sampling_params = SamplingParams(  
                max_tokens=kwargs.get("max_new_tokens", 512),  
                temperature=0.0 if not kwargs.get("do_sample", True) else kwargs.get("temperature", 0.7),  
                top_p=1.0 if not kwargs.get("do_sample", True) else kwargs.get("top_p", 0.9),  
                top_k=-1 if not kwargs.get("do_sample", True) else kwargs.get("top_k", 20),  
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),  
                presence_penalty=kwargs.get("presence_penalty", 0.0),  
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),  
                stop=kwargs.get("stop", ["<|im_end|>"]),  
                n=1,  
                best_of=kwargs.get("num_beams", 1) if not kwargs.get("do_sample", True) else 1  
            )  
            
            # 执行流式推理  
            current_length = 0  
            for output in self.model.generate(prompt, sampling_params, stream=True):  
                response = output.outputs[0].text  
                new_text = response[current_length:]  
                current_length = len(response)  
                yield new_text  
                
        except Exception as e:  
            logger.error(f"流式推理过程中出错: {str(e)}")  
            yield f"推理出错: {str(e)}"  
    
    def batch_stream_generate(self,   
                             prompts: List[str],   
                             **kwargs) -> Iterator[List[str]]:  
        """  
        批量流式生成文本  
        
        Args:  
            prompts: 输入提示列表  
            
        Yields:  
            当前时刻所有文本的新片段列表  
        """  
        try:  
            # 设置采样参数  
            sampling_params = SamplingParams(  
                max_tokens=kwargs.get("max_new_tokens", 512),  
                temperature=0.0 if not kwargs.get("do_sample", True) else kwargs.get("temperature", 0.7),  
                top_p=1.0 if not kwargs.get("do_sample", True) else kwargs.get("top_p", 0.9),  
                top_k=-1 if not kwargs.get("do_sample", True) else kwargs.get("top_k", 20),  
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),  
                presence_penalty=kwargs.get("presence_penalty", 0.0),  
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),  
                stop=kwargs.get("stop", ["<|im_end|>"]),  
                n=1,  
                best_of=kwargs.get("num_beams", 1) if not kwargs.get("do_sample", True) else 1  
            )  
            
            # 执行批量流式推理  
            current_lengths = [0] * len(prompts)  
            for outputs in self.model.generate(prompts, sampling_params, stream=True):  
                new_texts = []  
                for i, output in enumerate(outputs):  
                    response = output.outputs[0].text  
                    new_text = response[current_lengths[i]:]  
                    current_lengths[i] = len(response)  
                    new_texts.append(new_text)  
                yield new_texts  
                
        except Exception as e:  
            logger.error(f"批量流式推理过程中出错: {str(e)}")  
            yield [f"推理出错: {str(e)}"] * len(prompts)  
    
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
    if not VLLM_AVAILABLE:  
        print("请先安装vLLM库")  
        sys.exit(1)  
        
    # 测试模型路径，请替换为实际的模型路径  
    model_path = "/path/to/your/model"  
    
    # 初始化推理类  
    inference = VLLMInference(model_path)  
    
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
    
    # 测试批量生成  
    batch_queries = [  
        "高血压患者应该注意什么？",  
        "糖尿病的早期症状有哪些？",  
        "如何预防感冒？"  
    ]  
    
    batch_answers = inference.batch_generate([format_prompt(q) for q in batch_queries])  
    for q, a in zip(batch_queries, batch_answers):  
        print(f"问题: {q}")  
        print(f"回答: {a}")  
        print("-" * 50)  
    
    # 测试流式生成  
    print("流式生成:")  
    for text in inference.stream_generate(format_prompt(query)):  
        print(text, end="", flush=True)  
    print()  