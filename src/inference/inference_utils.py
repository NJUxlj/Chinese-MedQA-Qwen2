"""  
推理过程中需要用到的工具函数  
"""  
import os  
import sys  
import logging  
import json  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple  
import torch  
import numpy as np  

# 确保可以导入项目其他模块  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from config.model_config import ModelConfig  
from config.rag_config import RAGConfig  
from utils.logger import get_logger  

logger = get_logger("inference_utils")  

def format_prompt(query: str,   
                 history: Optional[List[Dict[str, str]]] = None,   
                 system_prompt: Optional[str] = None) -> str:  
    """  
    格式化提示词，将用户查询、历史对话和系统提示组合成模型输入格式  
    
    Args:  
        query: 用户查询  
        history: 历史对话记录，格式为[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]  
        system_prompt: 系统提示词  
        
    Returns:  
        格式化后的提示词  
    """  
    if history is None:  
        history = []  
        
    if system_prompt is None:  
        system_prompt = "你是一个专业的医疗助手，请根据用户的问题，提供准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
    
    # Qwen2模型的提示词格式  
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"  
    
    # 添加历史对话  
    for message in history:  
        role = message["role"]  
        content = message["content"]  
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"  
    
    # 添加当前用户查询  
    formatted_prompt += f"<|im_start|>user\n{query}<|im_end|>\n"  
    formatted_prompt += "<|im_start|>assistant\n"  
    
    return formatted_prompt  

def format_rag_prompt(query: str,   
                     context_docs: List[str],   
                     history: Optional[List[Dict[str, str]]] = None,   
                     system_prompt: Optional[str] = None) -> str:  
    """  
    格式化RAG提示词，将检索到的文档内容加入到提示词中  
    
    Args:  
        query: 用户查询  
        context_docs: 从知识库检索到的相关文档  
        history: 历史对话记录  
        system_prompt: 系统提示词  
        
    Returns:  
        格式化后的RAG提示词  
    """  
    if system_prompt is None:  
        system_prompt = "你是一个专业的医疗助手，请根据用户的问题和以下参考信息，提供准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
    
    # 整合上下文信息  
    context_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(context_docs)])  
    
    # 创建包含上下文的用户查询  
    context_query = f"问题：{query}\n\n参考信息：\n{context_text}\n\n请根据以上参考信息回答问题。"  
    
    # 使用普通的格式化函数处理带有上下文的查询  
    return format_prompt(context_query, history, system_prompt)  

def postprocess_response(response: str) -> str:  
    """  
    处理模型返回的响应，移除可能的特殊标记  
    
    Args:  
        response: 模型返回的原始响应  
        
    Returns:  
        处理后的响应  
    """  
    # 移除可能的结束标记  
    if "<|im_end|>" in response:  
        response = response.split("<|im_end|>")[0]  
    
    # 清理开头可能的助手标记  
    if response.startswith("assistant\n"):  
        response = response[len("assistant\n"):]  
    
    return response.strip()  

def measure_latency(func):  
    """  
    装饰器：测量函数的延迟时间  
    """  
    def wrapper(*args, **kwargs):  
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        latency = end_time - start_time  
        logger.info(f"Function {func.__name__} took {latency:.4f} seconds to execute.")  
        return result  
    return wrapper  

def get_model_path(model_name_or_path: str) -> str:  
    """  
    获取模型路径，支持本地路径或Hugging Face模型ID  
    
    Args:  
        model_name_or_path: 模型名称或路径  
        
    Returns:  
        完整的模型路径  
    """  
    # 如果是本地路径，直接返回  
    if os.path.exists(model_name_or_path):  
        return model_name_or_path  
    
    # 获取模型配置中的路径  
    model_config = ModelConfig()  
    model_dir = model_config.model_dir  
    
    # 检查是否在model_dir中  
    local_path = os.path.join(model_dir, model_name_or_path)  
    if os.path.exists(local_path):  
        return local_path  
    
    # 否则假设是Hugging Face模型ID  
    return model_name_or_path  

def text_to_tokens(text: str, tokenizer) -> List[int]:  
    """  
    将文本转换为token ID列表  
    
    Args:  
        text: 输入文本  
        tokenizer: 分词器  
        
    Returns:  
        token ID列表  
    """  
    return tokenizer(text).input_ids  

def tokens_to_text(tokens: List[int], tokenizer) -> str:  
    """  
    将token ID列表转换为文本  
    
    Args:  
        tokens: token ID列表  
        tokenizer: 分词器  
        
    Returns:  
        解码后的文本  
    """  
    return tokenizer.decode(tokens)  

def split_batch(texts: List[str], batch_size: int) -> List[List[str]]:  
    """  
    将文本列表拆分为批次  
    
    Args:  
        texts: 文本列表  
        batch_size: 批次大小  
        
    Returns:  
        批次列表  
    """  
    return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]  

def get_first_device():  
    """  
    获取第一个可用的GPU设备，如果没有则使用CPU  
    
    Returns:  
        设备字符串  
    """  
    if torch.cuda.is_available():  
        return f"cuda:{torch.cuda.current_device()}"  
    return "cpu"  

def check_cuda_memory():  
    """  
    检查CUDA内存使用情况  
    
    Returns:  
        Dict: 包含总内存和已用内存的字典  
    """  
    if not torch.cuda.is_available():  
        return {"total_memory": 0, "used_memory": 0}  
    
    device = torch.cuda.current_device()  
    total_memory = torch.cuda.get_device_properties(device).total_memory  
    reserved_memory = torch.cuda.memory_reserved(device)  
    allocated_memory = torch.cuda.memory_allocated(device)  
    
    return {  
        "total_memory": total_memory / (1024**3),  # GB  
        "reserved_memory": reserved_memory / (1024**3),  # GB  
        "allocated_memory": allocated_memory / (1024**3)  # GB  
    }  

def get_inference_params(config: Optional[Dict] = None) -> Dict:  
    """  
    获取推理参数  
    
    Args:  
        config: 自定义配置参数  
        
    Returns:  
        推理参数字典  
    """  
    default_params = {  
        "max_new_tokens": 512,  
        "temperature": 0.7,  
        "top_p": 0.9,  
        "top_k": 20,  
        "repetition_penalty": 1.1,  
        "do_sample": True,  
        "num_beams": 1  
    }  
    
    if config is not None:  
        default_params.update(config)  
    
    return default_params  