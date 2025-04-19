"""  
使用XInference进行模型推理的模块  
XInference是一个强大的模型推理引擎，支持多种大型语言模型的本地部署和推理  
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

logger = get_logger("xinference_inference")  

# 尝试导入XInference  
try:  
    from xinference.client import RESTfulClient  
    XINFERENCE_AVAILABLE = True  
except ImportError:  
    logger.warning("XInference库未安装，无法使用XInference进行推理。请使用以下命令安装：")  
    logger.warning("pip install 'xinference[all]'")  
    XINFERENCE_AVAILABLE = False  

class XInferenceInference:  
    """  
    基于XInference的推理类  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    
    def __new__(cls, model_uid: str, endpoint: str, *args, **kwargs):  
        """  
        实现单例模式，相同模型ID和端点只创建一个实例  
        """  
        with cls._lock:  
            instance_key = f"{model_uid}_{endpoint}"  
            if instance_key not in cls._instances:  
                instance = super(XInferenceInference, cls).__new__(cls)  
                cls._instances[instance_key] = instance  
            return cls._instances[instance_key]  
    
    def __init__(self,   
                model_uid: str,   
                endpoint: str = "http://localhost:9997",  
                model_type: str = "LLM"):  
        """  
        初始化XInference推理类  
        
        Args:  
            model_uid: 模型ID  
            endpoint: XInference服务器地址，默认为本地9997端口  
            model_type: 模型类型，可选LLM、TextEmbedding等  
        """  
        if not XINFERENCE_AVAILABLE:  
            raise ImportError("请先安装XInference库")  
        
        # 防止重复初始化  
        if hasattr(self, 'model_uid'):  
            return  
            
        self.model_uid = model_uid  
        self.endpoint = endpoint  
        self.model_type = model_type  
        self.client = None  
        self.model = None  
        self.model_description = None  
        self.context_length = None  
        
        logger.info(f"正在连接XInference服务: {self.endpoint}")  
        
        try:  
            # 创建RESTful客户端  
            self.client = RESTfulClient(self.endpoint)  
            
            # 获取模型信息  
            self._initialize_model()  
                
        except Exception as e:  
            logger.error(f"连接XInference服务时出错: {str(e)}")  
            logger.warning("请确保XInference服务已启动且可访问")  
    
    def _initialize_model(self):  
        """  
        初始化模型  
        """  
        try:  
            # 检查模型是否存在  
            models = self.client.list_models()  
            if self.model_uid not in models:  
                logger.warning(f"模型 {self.model_uid} 不存在，请先部署模型")  
                return  
            
            # 获取模型描述  
            self.model_description = models[self.model_uid]  
            logger.info(f"模型信息: {self.model_description}")  
            
            # 获取模型上下文窗口大小  
            if "context_length" in self.model_description:  
                self.context_length = self.model_description["context_length"]  
                
            # 获取模型实例  
            self.model = self.client.get_model(self.model_uid)  
            
            logger.info(f"模型 {self.model_uid} 初始化完成")  
            
        except Exception as e:  
            logger.error(f"初始化模型时出错: {str(e)}")  
    
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
    def chat(self,   
            messages: List[Dict[str, str]],   
            **kwargs) -> str:  
        """  
        对话模式生成文本  
        
        Args:  
            messages: 消息列表  
            
        Returns:  
            生成的回答  
        """  
        try:  
            if self.model is None:  
                self._initialize_model()  
                if self.model is None:  
                    return "模型未初始化"  
            
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备参数  
            params = {  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "max_tokens": inference_params.get("max_new_tokens", 512),  
            }  
            
            # XInference API支持的其他参数  
            if "top_k" in inference_params:  
                params["top_k"] = inference_params["top_k"]  
            if "repetition_penalty" in inference_params:  
                params["repetition_penalty"] = inference_params["repetition_penalty"]  
            if "presence_penalty" in inference_params:  
                params["presence_penalty"] = inference_params["presence_penalty"]  
            if "frequency_penalty" in inference_params:  
                params["frequency_penalty"] = inference_params["frequency_penalty"]  
            if "stop" in kwargs:  
                params["stop"] = kwargs["stop"]  
            
            # 执行推理  
            response = self.model.chat(messages=messages, **params)  
            
            # 提取生成的文本  
            if isinstance(response, dict):  
                generated_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')  
            else:  
                generated_text = str(response)  
            
            # 后处理  
            return postprocess_response(generated_text)  
            
        except Exception as e:  
            logger.error(f"推理过程中出错: {str(e)}")  
            return f"推理出错: {str(e)}"  
    
    @measure_latency  
    def generate(self,   
                prompt: str,   
                **kwargs) -> str:  
        """  
        生成文本  
        
        Args:  
            prompt: 输入提示  
            
        Returns:  
            生成的文本  
        """  
        try:  
            if self.model is None:  
                self._initialize_model()  
                if self.model is None:  
                    return "模型未初始化"  
            
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备参数  
            params = {  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "max_tokens": inference_params.get("max_new_tokens", 512),  
            }  
            
            # XInference API支持的其他参数  
            if "top_k" in inference_params:  
                params["top_k"] = inference_params["top_k"]  
            if "repetition_penalty" in inference_params:  
                params["repetition_penalty"] = inference_params["repetition_penalty"]  
            if "presence_penalty" in inference_params:  
                params["presence_penalty"] = inference_params["presence_penalty"]  
            if "frequency_penalty" in inference_params:  
                params["frequency_penalty"] = inference_params["frequency_penalty"]  
            if "stop" in kwargs:  
                params["stop"] = kwargs["stop"]  
            
            # 执行推理  
            response = self.model.generate(prompt=prompt, **params)  
            
            # 提取生成的文本  
            if isinstance(response, dict):  
                generated_text = response.get('choices', [{}])[0].get('text', '')  
            else:  
                generated_text = str(response)  
            
            # 后处理  
            return postprocess_response(generated_text)  
            
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
    
    def stream_chat(self,   
                   messages: List[Dict[str, str]],   
                   **kwargs) -> Iterator[str]:  
        """  
        流式对话模式生成文本  
        
        Args:  
            messages: 消息列表  
            
        Yields:  
            生成的文本片段  
        """  
        try:  
            if self.model is None:  
                self._initialize_model()  
                if self.model is None:  
                    yield "模型未初始化"  
                    return  
            
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备参数  
            params = {  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "max_tokens": inference_params.get("max_new_tokens", 512),  
            }  
            
            # XInference API支持的其他参数  
            if "top_k" in inference_params:  
                params["top_k"] = inference_params["top_k"]  
            if "repetition_penalty" in inference_params:  
                params["repetition_penalty"] = inference_params["repetition_penalty"]  
            if "presence_penalty" in inference_params:  
                params["presence_penalty"] = inference_params["presence_penalty"]  
            if "frequency_penalty" in inference_params:  
                params["frequency_penalty"] = inference_params["frequency_penalty"]  
            if "stop" in kwargs:  
                params["stop"] = kwargs["stop"]  
            
            # 执行流式推理  
            current_text = ""  
            for chunk in self.model.chat(messages=messages, stream=True, **params):  
                if isinstance(chunk, dict):  
                    chunk_text = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')  
                else:  
                    chunk_text = str(chunk)  
                
                if chunk_text:  
                    yield chunk_text  
                
        except Exception as e:  
            logger.error(f"流式推理过程中出错: {str(e)}")  
            yield f"推理出错: {str(e)}"  
    
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
            if self.model is None:  
                self._initialize_model()  
                if self.model is None:  
                    yield "模型未初始化"  
                    return  
            
            # 获取推理参数  
            inference_params = get_inference_params(kwargs)  
            
            # 准备参数  
            params = {  
                "temperature": inference_params.get("temperature", 0.7),  
                "top_p": inference_params.get("top_p", 0.9),  
                "max_tokens": inference_params.get("max_new_tokens", 512),  
            }  
            
            # XInference API支持的其他参数  
            if "top_k" in inference_params:  
                params["top_k"] = inference_params["top_k"]  
            if "repetition_penalty" in inference_params:  
                params["repetition_penalty"] = inference_params["repetition_penalty"]  
            if "presence_penalty" in inference_params:  
                params["presence_penalty"] = inference_params["presence_penalty"]  
            if "frequency_penalty" in inference_params:  
                params["frequency_penalty"] = inference_params["frequency_penalty"]  
            if "stop" in kwargs:  
                params["stop"] = kwargs["stop"]  
            
            # 执行流式推理  
            for chunk in self.model.generate(prompt=prompt, stream=True, **params):  
                if isinstance(chunk, dict):  
                    chunk_text = chunk.get('choices', [{}])[0].get('text', '')  
                else:  
                    chunk_text = str(chunk)  
                
                if chunk_text:  
                    yield chunk_text  
                
        except Exception as e:  
            logger.error(f"流式推理过程中出错: {str(e)}")  
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
    
    def embeddings(self, texts: List[str]) -> List[List[float]]:  
        """  
        获取文本的嵌入向量  
        
        Args:  
            texts: 文本列表  
            
        Returns:  
            嵌入向量列表  
        """  
        try:  
            if self.model is None:  
                self._initialize_model()  
                if self.model is None:  
                    return [[] for _ in range(len(texts))]  
            
            # 检查模型类型是否支持嵌入  
            if self.model_type != "TextEmbedding" and not hasattr(self.model, "embeddings"):  
                logger.warning(f"模型 {self.model_uid} 不支持嵌入功能")  
                return [[] for _ in range(len(texts))]  
            
            # 获取嵌入向量  
            embeddings_list = []  
            
            # 批量处理  
            for i in range(0, len(texts), 32):  # 每批32个文本  
                batch_texts = texts[i:i+32]  
                response = self.model.embeddings(texts=batch_texts)  
                
                if isinstance(response, dict) and "data" in response:  
                    for item in response["data"]:  
                        embeddings_list.append(item.get("embedding", []))  
                else:  
                    # 单个处理  
                    for text in batch_texts:  
                        try:  
                            single_response = self.model.embeddings(texts=[text])  
                            if isinstance(single_response, dict) and "data" in single_response:  
                                embeddings_list.append(single_response["data"][0].get("embedding", []))  
                            else:  
                                embeddings_list.append([])  
                        except Exception as e:  
                            logger.error(f"获取单个嵌入向量时出错: {str(e)}")  
                            embeddings_list.append([])  
            
            return embeddings_list  
            
        except Exception as e:  
            logger.error(f"获取嵌入向量时出错: {str(e)}")  
            # 返回空列表作为错误处理  
            return [[] for _ in range(len(texts))]  
    
    @staticmethod  
    def list_available_models(endpoint: str = "http://localhost:9997") -> Dict:  
        """  
        列出所有可用的模型  
        
        Args:  
            endpoint: XInference服务器地址  
            
        Returns:  
            可用模型字典  
        """  
        try:  
            client = RESTfulClient(endpoint)  
            models = client.list_models()  
            return models  
            
        except Exception as e:  
            logger.error(f"列出可用模型时出错: {str(e)}")  
            return {}  
    
    @staticmethod  
    def register_model(model_name: str,   
                      model_type: str = "LLM",   
                      model_size: str = "7b",   
                      quantization: Optional[str] = None,  
                      model_format: Optional[str] = None,  
                      model_path: Optional[str] = None,  
                      endpoint: str = "http://localhost:9997") -> str:  
        """  
        注册模型  
        
        Args:  
            model_name: 模型名称，如qwen、llama2、chatglm3等  
            model_type: 模型类型，LLM或TextEmbedding  
            model_size: 模型大小，如7b、14b等  
            quantization: 量化方式，如4bit、8bit等  
            model_format: 模型格式，如pytorch、ggmlv3等  
            model_path: 模型路径，本地模型或HuggingFace模型ID  
            endpoint: XInference服务器地址  
            
        Returns:  
            注册后的模型ID  
        """  
        try:  
            client = RESTfulClient(endpoint)  
            
            # 准备部署参数  
            params = {  
                "model_name": model_name,  
                "model_size": model_size  
            }  
            
            if quantization:  
                params["quantization"] = quantization  
            
            if model_format:  
                params["model_format"] = model_format  
            
            if model_path:  
                params["model_path"] = model_path  
            
            # 部署模型  
            if model_type == "LLM":  
                model_uid = client.launch_model(model_type="LLM", **params)  
            elif model_type == "TextEmbedding":  
                model_uid = client.launch_model(model_type="TextEmbedding", **params)  
            else:  
                raise ValueError(f"不支持的模型类型: {model_type}")  
            
            logger.info(f"模型 {model_name} 部署成功，模型ID: {model_uid}")  
            return model_uid  
            
        except Exception as e:  
            logger.error(f"注册模型时出错: {str(e)}")  
            return ""  
    
    @staticmethod  
    def terminate_model(model_uid: str, endpoint: str = "http://localhost:9997") -> bool:  
        """  
        终止模型  
        
        Args:  
            model_uid: 模型ID  
            endpoint: XInference服务器地址  
            
        Returns:  
            是否成功  
        """  
        try:  
            client = RESTfulClient(endpoint)  
            client.terminate_model(model_uid)  
            logger.info(f"模型 {model_uid} 已终止")  
            return True  
            
        except Exception as e:  
            logger.error(f"终止模型时出错: {str(e)}")  
            return False  


# 测试代码  
if __name__ == "__main__":  
    if not XINFERENCE_AVAILABLE:  
        print("请先安装XInference库")  
        sys.exit(1)  
        
    # 测试模型ID，请替换为实际的模型ID  
    model_uid = "your_model_uid_here"  # 需要替换为实际的模型ID  
    
    # 先列出所有可用的模型  
    print("列出所有可用模型...")  
    models = XInferenceInference.list_available_models()  
    if models:  
        print(f"可用模型: {json.dumps(models, indent=2, ensure_ascii=False)}")  
        
        # 检查指定的模型是否存在  
        if model_uid not in models:  
            print(f"模型 {model_uid} 不存在，请先部署模型或使用以下可用模型之一:")  
            for uid, info in models.items():  
                print(f"  - {uid}: {info.get('model_name', 'unknown')}")  
            
            # 可以选择部署一个模型  
            print("\n尝试部署一个新的模型...")  
            new_model_uid = XInferenceInference.register_model(  
                model_name="qwen",  
                model_size="7b",  
                quantization="4bit"  
            )  
            
            if new_model_uid:  
                print(f"新模型部署成功，模型ID: {new_model_uid}")  
                model_uid = new_model_uid  
            else:  
                print("模型部署失败，请检查XInference服务器状态")  
                sys.exit(1)  
    else:  
        print("无法获取可用模型列表，请检查XInference服务器状态")  
        sys.exit(1)  
    
    # 初始化推理类  
    try:  
        inference = XInferenceInference(model_uid)  
        
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
        
        # 测试嵌入（如果模型支持）  
        try:  
            print("\n测试嵌入:")  
            embeddings = inference.embeddings(["高血压是什么？"])  
            if embeddings[0]:  
                print(f"嵌入向量维度: {len(embeddings[0])}")  
            else:  
                print("该模型不支持嵌入功能")  
        except Exception as e:  
            print(f"嵌入测试失败: {str(e)}")  
        
    except Exception as e:  
        print(f"测试出错: {str(e)}")  