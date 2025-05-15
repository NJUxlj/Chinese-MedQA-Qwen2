"""  
使用Ollama进行模型推理的模块  
Ollama是一个轻量级的本地LLM推理引擎，可以轻松部署和运行大型语言模型  
"""  
import os  
import sys  
import time  
import json  
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

logger = get_logger("ollama_inference")  

# 尝试导入Ollama  
try:  
    import ollama  
    OLLAMA_AVAILABLE = True  
except ImportError:  
    logger.warning("Ollama库未安装，无法使用Ollama进行推理。请使用以下命令安装：")  
    logger.warning("pip install ollama")  
    OLLAMA_AVAILABLE = False  

class OllamaInference:  
    """  
    基于Ollama的推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, model_name: str, *args, **kwargs):  
        """  
        实现单例模式，相同模型名称只创建一个实例  
        """  
        with cls._lock:  
            if model_name not in cls._instances:  
                instance = super(OllamaInference, cls).__new__(cls)  
                cls._instances[model_name] = instance  
            return cls._instances[model_name]  
    
    def __init__(self,   
                model_name: str,   
                host: str = "http://localhost:11434",  
                keep_alive: str = "5m",  
                num_gpu: int = 1,  
                num_thread: Optional[int] = None):  
        """  
        初始化Ollama推理类  
        
        Args:  
            model_name: 模型名称  
            host: Ollama服务器地址，默认为本地11434端口  
            keep_alive: 模型保持活跃的时间  
            num_gpu: 使用的GPU数量  
            num_thread: 使用的线程数量  
        """  
        if not OLLAMA_AVAILABLE:  
            raise ImportError("请先安装Ollama库")  
        
        # 防止重复初始化  
        if hasattr(self, 'model_name'):  
            return  
            
        self.model_name = model_name  
        self.host = host  
        self.keep_alive = keep_alive  
        self.num_gpu = num_gpu  
        self.num_thread = num_thread or os.cpu_count()  
        
        # 设置Ollama客户端  
        ollama.client._host = self.host  
        
        logger.info(f"正在检查模型: {self.model_name}")  
        
        try:  
            # 检查模型是否存在  
            models = ollama.list()  
            model_exists = any(model.get('name') == self.model_name for model in models.get('models', []))  
            
            if not model_exists:  
                logger.warning(f"模型 {self.model_name} 不存在，请先使用 'ollama pull {self.model_name}' 下载模型")  
            else:  
                logger.info(f"模型 {self.model_name} 已存在")  
                
                # 获取模型信息  
                model_info = self._get_model_info()  
                logger.info(f"模型信息: {model_info}")  
                
        except Exception as e:  
            logger.error(f"检查模型时出错: {str(e)}")  
            logger.warning("请确保Ollama服务已启动且可访问")  
    
    def _get_model_info(self) -> Dict:  
        """  
        获取模型信息  
        
        Returns:  
            模型信息字典  
        """  
        try:  
            # 通过直接调用API获取更详细的模型信息  
            import requests  
            response = requests.get(f"{self.host}/api/show",   
                                   params={"name": self.model_name})  
            
            if response.status_code == 200:  
                return response.json()  
            else:  
                logger.warning(f"获取模型信息失败，状态码: {response.status_code}")  
                return {}  
                
        except Exception as e:  
            logger.error(f"获取模型信息时出错: {str(e)}")  
            return {}  
    
    def _prepare_messages(self,   
                         query: str,   
                         history: Optional[List[Dict[str, str]]] = None,  
                         system_prompt: Optional[str] = None) -> List[Dict[str, str]]:  
        """  
        准备消息格式  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Returns:  
            消息列表  
        """  
        messages = []  
        
        # 添加系统提示  
        if system_prompt:  
            messages.append({"role": "system", "content": system_prompt})  
        
        # 添加历史记录  
        if history:  
            for msg in history:  
                messages.append({"role": msg["role"], "content": msg["content"]})  
        
        # 添加当前查询  
        messages.append({"role": "user", "content": query})  
        
        return messages  
    
    @measure_latency  
    def generate(self,   
                prompt: str,   
                system: Optional[str] = None,  
                format: Optional[str] = None,  
                options: Optional[Dict] = None,  
                **kwargs) -> str:  
        """  
        使用非对话模式生成文本  
        
        Args:  
            prompt: 输入提示  
            system: 系统提示词  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Returns:  
            生成的文本  
        """  
        try:  
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备选项  
            if options is None:  
                options = {}  
            
            options.update({  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "top_k": inference_params.get("top_k", 40),  
                "num_predict": inference_params.get("max_new_tokens", 512),  
                "seed": kwargs.get("seed", 0),  
                "num_gpu": self.num_gpu,  
                "num_thread": self.num_thread,  
                "stop": kwargs.get("stop", []),  
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1)  
            })  
            
            # 执行推理  
            response = ollama.generate(  
                model=self.model_name,  
                prompt=prompt,  
                system=system,  
                format=format,  
                options=options,  
                keep_alive=self.keep_alive  
            )  
            
            # 提取生成的文本  
            generated_text = response.get('response', '')  
            
            # 后处理  
            return postprocess_response(generated_text)  
            
        except Exception as e:  
            logger.error(f"推理过程中出错: {str(e)}")  
            return f"推理出错: {str(e)}"  
    
    @measure_latency  
    def chat(self,   
            messages: List[Dict[str, str]],  
            format: Optional[str] = None,  
            options: Optional[Dict] = None,  
            **kwargs) -> str:  
        """  
        使用对话模式生成文本  
        
        Args:  
            messages: 消息列表  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Returns:  
            生成的回答  
        """  
        try:  
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备选项  
            if options is None:  
                options = {}  
            
            options.update({  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "top_k": inference_params.get("top_k", 40),  
                "num_predict": inference_params.get("max_new_tokens", 512),  
                "seed": kwargs.get("seed", 0),  
                "num_gpu": self.num_gpu,  
                "num_thread": self.num_thread,  
                "stop": kwargs.get("stop", []),  
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1)  
            })  
            
            # 执行推理  
            response = ollama.chat(  
                model=self.model_name,  
                messages=messages,  
                format=format,  
                options=options,  
                keep_alive=self.keep_alive  
            )  
            
            # 提取生成的文本  
            generated_text = response.get('message', {}).get('content', '')  
            
            # 后处理  
            return postprocess_response(generated_text)  
            
        except Exception as e:  
            logger.error(f"对话推理过程中出错: {str(e)}")  
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
        # 准备消息列表  
        messages = self._prepare_messages(query, history, system_prompt)  
        
        # 使用对话模式生成回答  
        return self.chat(messages, **kwargs)  
    
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
        # 设置RAG系统提示词  
        rag_system_prompt = system_prompt  
        if system_prompt is None:  
            rag_system_prompt = "你是一个专业的医疗助手，请根据用户的问题和提供的参考信息，给出准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
        
        # 整合上下文信息  
        context_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(context_docs)])  
        
        # 创建包含上下文的用户查询  
        context_query = f"问题：{query}\n\n参考信息：\n{context_text}\n\n请根据以上参考信息回答问题。"  
        
        # 准备消息列表  
        messages = self._prepare_messages(context_query, history, rag_system_prompt)  
        
        # 使用对话模式生成回答  
        return self.chat(messages, **kwargs)  
    
    def stream_generate(self,   
                       prompt: str,   
                       system: Optional[str] = None,  
                       format: Optional[str] = None,  
                       options: Optional[Dict] = None,  
                       **kwargs) -> Iterator[str]:  
        """  
        流式生成文本  
        
        Args:  
            prompt: 输入提示  
            system: 系统提示词  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Yields:  
            生成的文本片段  
        """  
        try:  
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备选项  
            if options is None:  
                options = {}  
            
            options.update({  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "top_k": inference_params.get("top_k", 40),  
                "num_predict": inference_params.get("max_new_tokens", 512),  
                "seed": kwargs.get("seed", 0),  
                "num_gpu": self.num_gpu,  
                "num_thread": self.num_thread,  
                "stop": kwargs.get("stop", []),  
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1)  
            })  
            
            # 执行流式推理  
            for chunk in ollama.generate(  
                model=self.model_name,  
                prompt=prompt,  
                system=system,  
                format=format,  
                options=options,  
                stream=True,  
                keep_alive=self.keep_alive  
            ):  
                yield chunk.get('response', '')  
                
        except Exception as e:  
            logger.error(f"流式推理过程中出错: {str(e)}")  
            yield f"推理出错: {str(e)}"  
    
    def stream_chat(self,   
                   messages: List[Dict[str, str]],  
                   format: Optional[str] = None,  
                   options: Optional[Dict] = None,  
                   **kwargs) -> Iterator[str]:  
        """  
        流式对话模式生成文本  
        
        Args:  
            messages: 消息列表  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Yields:  
            生成的文本片段  
        """  
        try:  
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备选项  
            if options is None:  
                options = {}  
            
            options.update({  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "top_k": inference_params.get("top_k", 40),  
                "num_predict": inference_params.get("max_new_tokens", 512),  
                "seed": kwargs.get("seed", 0),  
                "num_gpu": self.num_gpu,  
                "num_thread": self.num_thread,  
                "stop": kwargs.get("stop", []),  
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1)  
            })  
            
            # 执行流式推理  
            for chunk in ollama.chat(  
                model=self.model_name,  
                messages=messages,  
                format=format,  
                options=options,  
                stream=True,  
                keep_alive=self.keep_alive  
            ):  
                yield chunk.get('message', {}).get('content', '')  
                
        except Exception as e:  
            logger.error(f"流式对话推理过程中出错: {str(e)}")  
            yield f"推理出错: {str(e)}"  
    
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
        # 准备消息列表  
        messages = self._prepare_messages(query, history, system_prompt)  
        
        # 使用流式对话模式生成回答  
        yield from self.stream_chat(messages, **kwargs)  
    
    def stream_answer_with_rag(self,  
                            query: str,  
                            context_docs: List[str],  
                            history: Optional[List[Dict[str, str]]] = None,  
                            system_prompt: Optional[str] = None,  
                            **kwargs) -> Iterator[str]:  
        """  
        使用RAG技术流式回答问题  
        
        Args:  
            query: 用户查询  
            context_docs: 从知识库检索到的相关文档  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Yields:  
            生成的回答片段  
        """  
        # 设置RAG系统提示词  
        rag_system_prompt = system_prompt  
        if system_prompt is None:  
            rag_system_prompt = "你是一个专业的医疗助手，请根据用户的问题和提供的参考信息，给出准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
        
        # 整合上下文信息  
        context_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(context_docs)])  
        
        # 创建包含上下文的用户查询  
        context_query = f"问题：{query}\n\n参考信息：\n{context_text}\n\n请根据以上参考信息回答问题。"  
        
        # 准备消息列表  
        messages = self._prepare_messages(context_query, history, rag_system_prompt)  
        
        # 使用流式对话模式生成回答  
        yield from self.stream_chat(messages, **kwargs)  
    
    def embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:  
        """  
        获取文本的嵌入向量  
        
        Args:  
            texts: 文本列表  
            
        Returns:  
            嵌入向量列表  
        """  
        try:  
            # 确保文本是字符串列表  
            texts = [str(text) for text in texts]  
            
            # 获取嵌入向量  
            embeddings_list = []  
            for text in texts:  
                response = ollama.embeddings(  
                    model=self.model_name,  
                    prompt=text,  
                    keep_alive=self.keep_alive  
                )  
                embeddings_list.append(response.get('embedding', []))  
            
            return embeddings_list  
            
        except Exception as e:  
            logger.error(f"获取嵌入向量时出错: {str(e)}")  
            # 返回空列表作为错误处理  
            return [[] for _ in range(len(texts))]  
    
    def pull_model(self) -> bool:  
        """  
        拉取模型  
        
        Returns:  
            是否成功  
        """  
        try:  
            # 拉取模型  
            logger.info(f"正在拉取模型 {self.model_name}...")  
            response = ollama.pull(self.model_name)  
            
            logger.info(f"模型 {self.model_name} 拉取完成")  
            return True  
            
        except Exception as e:  
            logger.error(f"拉取模型时出错: {str(e)}")  
            return False  
    
    def is_model_available(self) -> bool:  
        """  
        检查模型是否可用  
        
        Returns:  
            模型是否可用  
        """  
        try:  
            # 检查模型是否存在  
            models = ollama.list()  
            return any(model.get('name') == self.model_name for model in models.get('models', []))  
            
        except Exception as e:  
            logger.error(f"检查模型时出错: {str(e)}")  
            return False  


# 测试代码  
if __name__ == "__main__":  
    if not OLLAMA_AVAILABLE:  
        print("请先安装Ollama库")  
        sys.exit(1)  
        
    # 测试模型名称，请替换为实际的模型名称  
    model_name = "qwen:7b" # 或其他支持的模型  
    
    # 检查模型是否已本地下载  
    try:  
        models = ollama.list()  
        model_exists = any(model.get('name') == model_name for model in models.get('models', []))  
        
        if not model_exists:  
            print(f"模型 {model_name} 不存在，请先使用 'ollama pull {model_name}' 下载模型")  
            sys.exit(1)  
    except Exception as e:  
        print(f"检查模型时出错: {str(e)}")  
        print("请确保Ollama服务已启动且可访问")  
        sys.exit(1)  
    
    # 初始化推理类  
    inference = OllamaInference(model_name)  
    
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
    print("\n流式生成:")  
    for text in inference.stream_answer(query):  
        print(text, end="", flush=True)  
    print()  
    
    # 测试嵌入  
    print("\n测试嵌入:")  
    embeddings = inference.embeddings(["高血压是什么？"])  
    print(f"嵌入向量维度: {len(embeddings[0])}")  