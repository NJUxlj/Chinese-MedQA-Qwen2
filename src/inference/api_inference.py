"""  
使用API进行模型推理的模块  
支持智谱、百度文心、OpenAI等API模型的调用  
"""  
import os  
import sys  
import time  
import json  
import requests  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator  
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
    get_inference_params  
)  

logger = get_logger("api_inference")  

# 尝试导入API库  
try:  
    import openai  
    OPENAI_AVAILABLE = True  
except ImportError:  
    logger.warning("openai库未安装，无法使用OpenAI API。请使用以下命令安装：")  
    logger.warning("pip install openai")  
    OPENAI_AVAILABLE = False  

try:  
    from zhipuai import ZhipuAI  
    ZHIPUAI_AVAILABLE = True  
except ImportError:  
    logger.warning("zhipuai库未安装，无法使用智谱API。请使用以下命令安装：")  
    logger.warning("pip install zhipuai")  
    ZHIPUAI_AVAILABLE = False  

try:  
    from dashscope import Generation  
    DASHSCOPE_AVAILABLE = True  
except ImportError:  
    logger.warning("dashscope库未安装，无法使用阿里云API。请使用以下命令安装：")  
    logger.warning("pip install dashscope")  
    DASHSCOPE_AVAILABLE = False  


class APIInference:  
    """  
    API推理基类  
    """  
    def __init__(self, api_key: str, api_base: Optional[str] = None):  
        """  
        初始化API推理类  
        
        Args:  
            api_key: API密钥  
            api_base: API基础URL  
        """  
        self.api_key = api_key  
        self.api_base = api_base  
    
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
        raise NotImplementedError("子类必须实现此方法")  
    
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
        # 格式化RAG提示词，然后使用普通方法回答  
        rag_system_prompt = system_prompt  
        if system_prompt is None:  
            rag_system_prompt = "你是一个专业的医疗助手，请根据用户的问题和提供的参考信息，给出准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
        
        # 整合上下文信息  
        context_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(context_docs)])  
        
        # 创建包含上下文的用户查询  
        context_query = f"问题：{query}\n\n参考信息：\n{context_text}\n\n请根据以上参考信息回答问题。"  
        
        # 使用普通问答方法  
        return self.answer_question(context_query, history, rag_system_prompt, **kwargs)  
    
    def stream_answer(self,   
                     query: str,   
                     history: Optional[List[Dict[str, str]]] = None,  
                     system_prompt: Optional[str] = None,  
                     **kwargs) -> Iterator[str]:  
        """  
        流式回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Yields:  
            生成的回答片段  
        """  
        raise NotImplementedError("子类必须实现此方法")  


class ZhipuAIInference(APIInference):  
    """  
    智谱AI API推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, api_key: str, *args, **kwargs):  
        """  
        实现单例模式，相同API密钥只创建一个实例  
        """  
        with cls._lock:  
            if api_key not in cls._instances:  
                instance = super(ZhipuAIInference, cls).__new__(cls)  
                cls._instances[api_key] = instance  
            return cls._instances[api_key]  
    
    def __init__(self, api_key: str, model: str = "glm-4"):  
        """  
        初始化智谱API推理类  
        
        Args:  
            api_key: API密钥  
            model: 模型名称  
        """  
        if not ZHIPUAI_AVAILABLE:  
            raise ImportError("请先安装zhipuai库")  
        
        # 防止重复初始化  
        if hasattr(self, 'client'):  
            return  
            
        super().__init__(api_key)  
        self.model = model  
        self.client = ZhipuAI(api_key=api_key)  
        
        logger.info(f"智谱AI API初始化完成，使用模型: {model}")  
    
    @measure_latency  
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
        try:  
            if history is None:  
                history = []  
                
            # 转换为智谱API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512)  
            )  
            
            return response.choices[0].message.content  
            
        except Exception as e:  
            logger.error(f"智谱API调用出错: {str(e)}")  
            return f"API调用出错: {str(e)}"  
    
    def stream_answer(self,   
                     query: str,   
                     history: Optional[List[Dict[str, str]]] = None,  
                     system_prompt: Optional[str] = None,  
                     **kwargs) -> Iterator[str]:  
        """  
        流式回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Yields:  
            生成的回答片段  
        """  
        try:  
            if history is None:  
                history = []  
                
            # 转换为智谱API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512),  
                stream=True  
            )  
            
            for chunk in response:  
                if chunk.choices and chunk.choices[0].delta.content:  
                    yield chunk.choices[0].delta.content  
            
        except Exception as e:  
            logger.error(f"智谱API流式调用出错: {str(e)}")  
            yield f"API调用出错: {str(e)}"  


class DashScopeInference(APIInference):  
    """  
    阿里云灵积API推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, api_key: str, *args, **kwargs):  
        """  
        实现单例模式，相同API密钥只创建一个实例  
        """  
        with cls._lock:  
            if api_key not in cls._instances:  
                instance = super(DashScopeInference, cls).__new__(cls)  
                cls._instances[api_key] = instance  
            return cls._instances[api_key]  
    
    def __init__(self, api_key: str, model: str = "qwen-max"):  
        """  
        初始化阿里云灵积API推理类  
        
        Args:  
            api_key: API密钥  
            model: 模型名称  
        """  
        if not DASHSCOPE_AVAILABLE:  
            raise ImportError("请先安装dashscope库")  
        
        # 防止重复初始化  
        if hasattr(self, 'generation'):  
            return  
            
        super().__init__(api_key)  
        self.model = model  
        self.generation = Generation()  
        # 设置环境变量  
        os.environ['DASHSCOPE_API_KEY'] = api_key  
        
        logger.info(f"阿里云灵积API初始化完成，使用模型: {model}")  
    
    @measure_latency  
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
        try:  
            if history is None:  
                history = []  
                
            # 转换为灵积API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.generation.call(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512),  
                result_format='message'  
            )  
            
            if response.status_code == 200:  
                content = response.output.choices[0]['message']['content']  
                return content  
            else:  
                logger.error(f"灵积API返回错误: {response.code}, {response.message}")  
                return f"API调用出错: {response.message}"  
            
        except Exception as e:  
            logger.error(f"灵积API调用出错: {str(e)}")  
            return f"API调用出错: {str(e)}"  
    
    def stream_answer(self,   
                     query: str,   
                     history: Optional[List[Dict[str, str]]] = None,  
                     system_prompt: Optional[str] = None,  
                     **kwargs) -> Iterator[str]:  
        """  
        流式回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Yields:  
            生成的回答片段  
        """  
        try:  
            if history is None:  
                history = []  
                
            # 转换为灵积API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.generation.call(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512),  
                result_format='message',  
                stream=True  
            )  
            
            for chunk in response:  
                if chunk.status_code == 200:  
                    if chunk.output and chunk.output.choices and chunk.output.choices[0].get('message', {}).get('content'):  
                        yield chunk.output.choices[0]['message']['content']  
                else:  
                    logger.error(f"灵积API流式返回错误: {chunk.code}, {chunk.message}")  
                    yield f"API调用出错: {chunk.message}"  
            
        except Exception as e:  
            logger.error(f"灵积API流式调用出错: {str(e)}")  
            yield f"API调用出错: {str(e)}"  


class OpenAIInference(APIInference):  
    """  
    OpenAI API推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, api_key: str, *args, **kwargs):  
        """  
        实现单例模式，相同API密钥只创建一个实例  
        """  
        with cls._lock:  
            instance_key = f"{api_key}_{kwargs.get('api_base', 'default')}"  
            if instance_key not in cls._instances:  
                instance = super(OpenAIInference, cls).__new__(cls)  
                cls._instances[instance_key] = instance  
            return cls._instances[instance_key]  
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", api_base: Optional[str] = None):  
        """  
        初始化OpenAI API推理类  
        
        Args:  
            api_key: API密钥  
            model: 模型名称  
            api_base: API基础URL，可用于自定义API端点  
        """  
        if not OPENAI_AVAILABLE:  
            raise ImportError("请先安装openai库")  
        
        # 防止重复初始化  
        if hasattr(self, 'client'):  
            return  
            
        super().__init__(api_key, api_base)  
        self.model = model  
        
        # 初始化客户端  
        if api_base:  
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)  
        else:  
            self.client = openai.OpenAI(api_key=api_key)  
        
        logger.info(f"OpenAI API初始化完成，使用模型: {model}")  
    
    @measure_latency  
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
        try:  
            if history is None:  
                history = []  
                
            # 转换为OpenAI API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512)  
            )  
            
            return response.choices[0].message.content  
            
        except Exception as e:  
            logger.error(f"OpenAI API调用出错: {str(e)}")  
            return f"API调用出错: {str(e)}"  
    
    def stream_answer(self,   
                     query: str,   
                     history: Optional[List[Dict[str, str]]] = None,  
                     system_prompt: Optional[str] = None,  
                     **kwargs) -> Iterator[str]:  
        """  
        流式回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Yields:  
            生成的回答片段  
        """  
        try:  
            if history is None:  
                history = []  
                
            # 转换为OpenAI API的历史记录格式  
            messages = []  
            
            # 添加系统提示  
            if system_prompt:  
                messages.append({"role": "system", "content": system_prompt})  
            
            # 添加历史记录  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
            
            # 添加当前查询  
            messages.append({"role": "user", "content": query})  
            
            # 获取生成参数  
            inference_params = get_inference_params(kwargs)  
            
            # 调用API  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=messages,  
                temperature=inference_params.get("temperature", 0.7),  
                top_p=inference_params.get("top_p", 0.9),  
                max_tokens=inference_params.get("max_new_tokens", 512),  
                stream=True  
            )  
            
            for chunk in response:  
                if chunk.choices and chunk.choices[0].delta.content:  
                    yield chunk.choices[0].delta.content  
            
        except Exception as e:  
            logger.error(f"OpenAI API流式调用出错: {str(e)}")  
            yield f"API调用出错: {str(e)}"  


# 工厂方法，根据类型创建不同的推理对象  
def create_inference(inference_type: str, **kwargs):  
    """  
    根据类型创建推理对象  
    
    Args:  
        inference_type: 推理类型，支持fastllm、vllm、zhipuai、dashscope、openai  
        **kwargs: 其他参数  
        
    Returns:  
        推理对象  
    """  
    if inference_type == "fastllm":  
        from inference.fastllm_inference import FastLLMInference, FASTLLM_AVAILABLE  
        if not FASTLLM_AVAILABLE:  
            raise ImportError("请先安装fastllm库")  
        return FastLLMInference(**kwargs)  
    
    elif inference_type == "vllm":  
        from inference.vllm_inference import VLLMInference, VLLM_AVAILABLE  
        if not VLLM_AVAILABLE:  
            raise ImportError("请先安装vllm库")  
        return VLLMInference(**kwargs)  
    
    elif inference_type == "zhipuai":  
        if not ZHIPUAI_AVAILABLE:  
            raise ImportError("请先安装zhipuai库")  
        return ZhipuAIInference(**kwargs)  
    
    elif inference_type == "dashscope":  
        if not DASHSCOPE_AVAILABLE:  
            raise ImportError("请先安装dashscope库")  
        return DashScopeInference(**kwargs)  
    
    elif inference_type == "openai":  
        if not OPENAI_AVAILABLE:  
            raise ImportError("请先安装openai库")  
        return OpenAIInference(**kwargs)  
    
    else:  
        raise ValueError(f"不支持的推理类型: {inference_type}")  


# 测试代码  
if __name__ == "__main__":  
    # 这里只是展示API的使用方法，实际测试需要提供有效的API密钥  
    # ZhipuAI测试  
    if ZHIPUAI_AVAILABLE:  
        api_key = "your_zhipuai_api_key"  # 替换为实际的API密钥  
        try:  
            inference = ZhipuAIInference(api_key)  
            query = "高血压患者应该注意什么？"  
            print("问题:", query)  
            
            answer = inference.answer_question(query)  
            print("智谱AI回答:", answer)  
            
            # 测试流式生成  
            print("智谱AI流式回答:")  
            for text in inference.stream_answer(query):  
                print(text, end="", flush=True)  
            print()  
        except Exception as e:  
            print(f"智谱AI API测试失败: {str(e)}")  
    
    # DashScope测试  
    if DASHSCOPE_AVAILABLE:  
        api_key = "your_dashscope_api_key"  # 替换为实际的API密钥  
        try:  
            inference = DashScopeInference(api_key)  
            query = "高血压患者应该注意什么？"  
            print("问题:", query)  
            
            answer = inference.answer_question(query)  
            print("灵积API回答:", answer)  
            
            # 测试流式生成  
            print("灵积API流式回答:")  
            for text in inference.stream_answer(query):  
                print(text, end="", flush=True)  
            print()  
        except Exception as e:  
            print(f"灵积API测试失败: {str(e)}")  
    
    # OpenAI测试  
    if OPENAI_AVAILABLE:  
        api_key = "your_openai_api_key"  # 替换为实际的API密钥  
        try:  
            inference = OpenAIInference(api_key)  
            query = "高血压患者应该注意什么？"  
            print("问题:", query)  
            
            answer = inference.answer_question(query)  
            print("OpenAI回答:", answer)  
            
            # 测试流式生成  
            print("OpenAI流式回答:")  
            for text in inference.stream_answer(query):  
                print(text, end="", flush=True)  
            print()  
        except Exception as e:  
            print(f"OpenAI API测试失败: {str(e)}")  