"""  
使用Ollama进行模型推理的模块  
Ollama是一个轻量级的本地LLM推理引擎，可以轻松部署和运行大型语言模型  
"""  
import os  
import sys  
import time  
import json  
import psutil
import threading
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator, Generator  
from threading import Lock  
from contextlib import contextmanager
from dataclasses import dataclass
from collections import deque

# 确保可以导入项目其他模块  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from utils.logger import setup_logger  
from config.model_config import ModelConfig  
from inference.inference_utils import (  
    format_prompt,   
    format_rag_prompt,   
    postprocess_response,   
    measure_latency,  
    get_inference_params  
)  

logger = setup_logger("ollama_inference", level="INFO")  

# 尝试导入Ollama  
try:  
    import ollama  
except ImportError:  
    raise ImportError("Ollama库未安装，无法使用Ollama进行推理。请使用以下命令安装：pip install ollama")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil未安装，将无法监控GPU使用情况")

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    avg_latency: float = 0.0
    gpu_memory_used: List[float] = None
    cpu_usage: float = 0.0
    
    def __post_init__(self):
        if self.gpu_memory_used is None:
            self.gpu_memory_used = []
    
    def to_dict(self) -> Dict:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'avg_latency': self.avg_latency,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'gpu_memory_used': self.gpu_memory_used,
            'cpu_usage': self.cpu_usage
        }  

class OllamaInference:  
    """  
    基于Ollama的推理类 - 优化版  
    """  
    _instances = {}  # 类实例字典，用于实现单例模式  
    _lock = Lock()  # 用于线程安全的锁  
    _request_times = deque(maxlen=100)  # 最近100次请求时间，用于计算平均延迟
    
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
                keep_alive: str = "10m",  # 增加默认保持时间  
                num_gpu: Optional[int] = None,  # 改为可选，让系统自动检测  
                num_thread: Optional[int] = None,  
                enable_monitoring: bool = True):  
        """  
        初始化Ollama推理类  
        
        Args:  
            model_name: 模型名称  
            host: Ollama服务器地址，默认为本地11434端口  
            keep_alive: 模型保持活跃的时间，默认10分钟  
            num_gpu: 使用的GPU数量，None表示自动检测  
            num_thread: 使用的线程数量，None表示自动检测  
            enable_monitoring: 是否启用性能监控  
        """  
        
        # 防止重复初始化  
        if hasattr(self, 'model_name'):  
            return  
            
        self.model_name = model_name  
        self.host = host  
        self.keep_alive = keep_alive  
        self.enable_monitoring = enable_monitoring
        self.performance_metrics = PerformanceMetrics()
        
        # 自动检测GPU配置
        if num_gpu is None:
            self.num_gpu = self._auto_detect_gpu_count()
        else:
            self.num_gpu = min(num_gpu, self._get_available_gpu_count())
        
        # 自动检测线程数
        if num_thread is None:
            self.num_thread = self._auto_detect_thread_count()
        else:
            self.num_thread = num_thread
        
        # 初始化客户端
        self._client = None
        self._initialize_client()
        
        # 性能监控
        self._monitoring_thread = None
        if self.enable_monitoring:
            self._start_monitoring()
        
        # 检查模型
        self._check_model()
        
        logger.info(f"Ollama推理器初始化完成 - 模型: {self.model_name}, GPU: {self.num_gpu}, 线程: {self.num_thread}")  
    
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
    
    def _auto_detect_gpu_count(self) -> int:
        """自动检测可用的GPU数量"""
        if not GPU_AVAILABLE:
            logger.info("GPU监控不可用，使用CPU模式")
            return 0
        
        try:
            gpus = GPUtil.getGPUs()
            available_gpus = [gpu for gpu in gpus if gpu.load < 0.9]  # 负载小于90%的GPU
            if available_gpus:
                count = min(len(available_gpus), 4)  # 最多使用4个GPU
                logger.info(f"检测到 {len(gpus)} 个GPU，可用 {len(available_gpus)} 个，将使用 {count} 个")
                return count
            else:
                logger.info("所有GPU负载过高，使用CPU模式")
                return 0
        except Exception as e:
            logger.error(f"检测GPU时出错: {e}")
            return 0
    
    def _get_available_gpu_count(self) -> int:
        """获取可用GPU数量"""
        if not GPU_AVAILABLE:
            return 0
        
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus)
        except Exception as e:
            logger.error(f"获取GPU数量时出错: {e}")
            return 0
    
    def _auto_detect_thread_count(self) -> int:
        """自动检测合适的线程数"""
        cpu_count = os.cpu_count()
        # 对于GPU推理，使用较少的线程；对于CPU推理，使用更多线程
        if self.num_gpu > 0:
            # GPU推理：使用CPU核心数的1/2，但至少4个线程
            return max(4, cpu_count // 2)
        else:
            # CPU推理：使用75%的CPU核心，但至少8个线程
            return max(8, int(cpu_count * 0.75))
    
    def _initialize_client(self):
        """初始化客户端连接"""
        try:
            self._client = ollama.Client(host=self.host)
            logger.info(f"客户端初始化成功 - {self.host}")
        except Exception as e:
            logger.error(f"初始化客户端失败: {e}")
            self._client = None
    
    def _start_monitoring(self):
        """启动性能监控"""
        def monitor():
            while hasattr(self, 'enable_monitoring') and self.enable_monitoring:
                try:
                    # 更新CPU使用率
                    self.performance_metrics.cpu_usage = psutil.cpu_percent()
                    
                    # 更新GPU内存使用
                    if GPU_AVAILABLE:
                        gpu_memory = []
                        for gpu in GPUtil.getGPUs():
                            gpu_memory.append(gpu.memoryUsed)
                        self.performance_metrics.gpu_memory_used = gpu_memory
                    
                    time.sleep(30)  # 每30秒更新一次
                except Exception as e:
                    logger.error(f"性能监控出错: {e}")
                    time.sleep(30)
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
        logger.info("性能监控已启动")
    
    def _check_model(self):
        """检查模型是否存在"""
        if not self._client:
            logger.error("客户端未初始化，无法检查模型")
            return
        
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
    
    def _update_metrics(self, success: bool, latency: float, tokens_generated: int = 0):
        """更新性能指标"""
        self.performance_metrics.total_requests += 1
        self._request_times.append(latency)
        
        if success:
            self.performance_metrics.successful_requests += 1
            self.performance_metrics.total_tokens_generated += tokens_generated
        else:
            self.performance_metrics.failed_requests += 1
        
        # 计算平均延迟
        if self._request_times:
            self.performance_metrics.avg_latency = sum(self._request_times) / len(self._request_times)
    
    @contextmanager
    def _request_context(self):
        """请求上下文管理器，用于资源管理"""
        start_time = time.time()
        success = False
        tokens_generated = 0
        
        try:
            yield start_time, success, tokens_generated
        finally:
            end_time = time.time()
            latency = end_time - start_time
            self._update_metrics(success, latency, tokens_generated)  
    
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
        使用非对话模式生成文本 - 优化版
        
        Args:  
            prompt: 输入提示  
            system: 系统提示词  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Returns:  
            生成的文本  
        """  
        if not self._client:
            return "错误：客户端未初始化"
        
        with self._request_context() as (start_time, success, tokens_generated):
            try:  
                # 获取推理参数  
                inference_params = get_inference_params(kwargs)  
                
                # 准备选项 - 优化参数设置
                if options is None:
                    options = {}
                
                # 动态调整参数基于模型大小和可用资源
                base_temp = inference_params.get("temperature", 0.7)
                base_top_p = inference_params.get("top_p", 0.9)
                base_top_k = inference_params.get("top_k", 40)
                
                # GPU推理可以使用更多参数优化
                if self.num_gpu > 0:
                    options.update({
                        "temperature": base_temp,
                        "top_p": base_top_p,
                        "top_k": base_top_k,
                        "num_predict": inference_params.get("max_new_tokens", 512),
                        "seed": kwargs.get("seed", 42),  # 使用固定种子提高可重复性
                        "num_gpu": self.num_gpu,
                        "num_thread": self.num_thread,
                        "stop": kwargs.get("stop", []),
                        "repeat_penalty": inference_params.get("repetition_penalty", 1.1),
                        "mirostat": 2,  # 启用自适应Mirostat
                        "mirostat_eta": 0.1,  # 学习率
                        "mirostat_tau": 5.0,  # 目标复杂度
                        "num_ctx": 4096,  # 上下文长度
                    })
                else:
                    # CPU推理参数优化
                    options.update({
                        "temperature": min(base_temp, 0.8),  # 降低温度以减少计算量
                        "top_p": min(base_top_p, 0.95),
                        "top_k": min(base_top_k, 50),
                        "num_predict": min(inference_params.get("max_new_tokens", 256), 512),
                        "seed": kwargs.get("seed", 42),
                        "num_thread": self.num_thread,
                        "stop": kwargs.get("stop", []),
                        "repeat_penalty": inference_params.get("repetition_penalty", 1.1),
                        "num_ctx": 2048,  # 较小的上下文以节省内存
                    })
                
                # 验证模型可用性
                if not self.is_model_available():
                    logger.error(f"模型 {self.model_name} 不可用")
                    return "错误：模型不可用"
                
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
                tokens_generated = len(generated_text.split())  # 粗略估计token数
                
                # 标记成功
                success = True
                
                # 后处理  
                processed_text = postprocess_response(generated_text)
                logger.debug(f"生成完成 - Token数: {tokens_generated}, 延迟: {time.time() - start_time:.2f}s")
                return processed_text
                
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
        使用对话模式生成文本 - 优化版
        
        Args:  
            messages: 消息列表  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Returns:  
            生成的回答  
        """  
        if not self._client:
            return "错误：客户端未初始化"
        
        if not messages:
            return "错误：消息列表为空"
        
        with self._request_context() as (start_time, success, tokens_generated):
            try:  
                # 获取推理参数  
                inference_params = get_inference_params(kwargs)  
                
                # 准备选项 - 优化参数设置
                if options is None:
                    options = {}
                
                # 复用generate方法的参数优化逻辑
                self._apply_optimized_options(options, inference_params, kwargs)
                
                # 验证模型可用性
                if not self.is_model_available():
                    logger.error(f"模型 {self.model_name} 不可用")
                    return "错误：模型不可用"
                
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
                tokens_generated = len(generated_text.split())
                
                # 标记成功
                success = True
                
                # 后处理  
                processed_text = postprocess_response(generated_text)
                logger.debug(f"对话生成完成 - Token数: {tokens_generated}, 延迟: {time.time() - start_time:.2f}s")
                return processed_text
                
            except Exception as e:  
                logger.error(f"对话推理过程中出错: {str(e)}")  
                return f"推理出错: {str(e)}"  
    
    def _apply_optimized_options(self, options: Dict, inference_params: Dict, kwargs: Dict):
        """应用优化的推理参数"""
        base_temp = inference_params.get("temperature", 0.7)
        base_top_p = inference_params.get("top_p", 0.9)
        base_top_k = inference_params.get("top_k", 40)
        
        # GPU推理可以使用更多参数优化
        if self.num_gpu > 0:
            options.update({
                "temperature": base_temp,
                "top_p": base_top_p,
                "top_k": base_top_k,
                "num_predict": inference_params.get("max_new_tokens", 512),
                "seed": kwargs.get("seed", 42),
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "stop": kwargs.get("stop", []),
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1),
                "mirostat": 2,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
                "num_ctx": 4096,
            })
        else:
            # CPU推理参数优化
            options.update({
                "temperature": min(base_temp, 0.8),
                "top_p": min(base_top_p, 0.95),
                "top_k": min(base_top_k, 50),
                "num_predict": min(inference_params.get("max_new_tokens", 256), 512),
                "seed": kwargs.get("seed", 42),
                "num_thread": self.num_thread,
                "stop": kwargs.get("stop", []),
                "repeat_penalty": inference_params.get("repetition_penalty", 1.1),
                "num_ctx": 2048,
            })  
    
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
                       **kwargs) -> Generator[str, None, None]:  
        """  
        流式生成文本 - 优化版
        
        Args:  
            prompt: 输入提示  
            system: 系统提示词  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Yields:  
            生成的文本片段  
        """  
        if not self._client:
            yield "错误：客户端未初始化"
            return
        
        with self._request_context() as (start_time, success, tokens_generated):
            try:  
                # 获取推理参数  
                inference_params = get_inference_params(kwargs)  
                
                # 准备选项  
                if options is None:
                    options = {}
                
                # 应用优化参数
                self._apply_optimized_options(options, inference_params, kwargs)
                
                # 验证模型可用性
                if not self.is_model_available():
                    yield "错误：模型不可用"
                    return
                
                total_generated = ""
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
                    chunk_content = chunk.get('response', '')
                    if chunk_content:
                        total_generated += chunk_content
                        success = True
                        yield chunk_content
                
                # 更新生成的token数
                tokens_generated = len(total_generated.split())
                
            except Exception as e:  
                logger.error(f"流式推理过程中出错: {str(e)}")  
                yield f"推理出错: {str(e)}"  
    
    def stream_chat(self,   
                   messages: List[Dict[str, str]],  
                   format: Optional[str] = None,   
                   options: Optional[Dict] = None,  
                   **kwargs) -> Generator[str, None, None]:  
        """  
        流式对话模式生成文本 - 优化版
        
        Args:  
            messages: 消息列表  
            format: 输出格式，如'json'  
            options: 额外选项  
            
        Yields:  
            生成的文本片段  
        """  
        if not self._client:
            yield "错误：客户端未初始化"
            return
        
        if not messages:
            yield "错误：消息列表为空"
            return
        
        with self._request_context() as (start_time, success, tokens_generated):
            try:  
                # 获取推理参数  
                inference_params = get_inference_params(kwargs)  
                
                # 准备选项  
                if options is None:
                    options = {}
                
                # 应用优化参数
                self._apply_optimized_options(options, inference_params, kwargs)
                
                # 验证模型可用性
                if not self.is_model_available():
                    yield "错误：模型不可用"
                    return
                
                total_generated = ""
                # 执行流式推理  
                for chunk in ollama.chat(  
                    model=self.model_name,  
                    messages=messages,  
                    format=format,  
                    options=options,  
                    stream=True,  
                    keep_alive=self.keep_alive  
                ):  
                    chunk_content = chunk.get('message', {}).get('content', '')
                    if chunk_content:
                        total_generated += chunk_content
                        success = True
                        yield chunk_content
                
                # 更新生成的token数
                tokens_generated = len(total_generated.split())
                
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
        if not self._client:
            return False
        
        try:  
            # 检查模型是否存在  
            models = ollama.list()  
            return any(model.get('name') == self.model_name for model in models.get('models', []))  
            
        except Exception as e:  
            logger.error(f"检查模型时出错: {str(e)}")  
            return False
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return self.performance_metrics.to_dict()
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        status = {
            "model_available": self.is_model_available(),
            "client_initialized": self._client is not None,
            "performance_metrics": self.get_performance_metrics(),
            "gpu_count": self.num_gpu,
            "thread_count": self.num_thread,
            "monitoring_enabled": self.enable_monitoring
        }
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                status["gpu_info"] = [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
                    for gpu in gpus
                ]
            except Exception as e:
                status["gpu_info"] = {"error": str(e)}
        
        return status
    
    def cleanup(self):
        """清理资源"""
        self.enable_monitoring = False
        if hasattr(self, '_monitoring_thread') and self._monitoring_thread:
            self._monitoring_thread = None
        
        if hasattr(self, '_client') and self._client:
            self._client = None
        
        logger.info("Ollama推理器资源已清理")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
    
    @staticmethod
    def create_optimal_config(model_name: str, 
                            gpu_available: bool = None,
                            memory_gb: float = None) -> Dict:
        """创建优化的配置"""
        config = {
            "model_name": model_name,
            "host": "http://localhost:11434",
            "keep_alive": "10m",
            "enable_monitoring": True
        }
        
        # GPU配置
        if gpu_available is None:
            gpu_available = GPU_AVAILABLE
        
        if gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # 选择负载最低的GPU
                    best_gpu = min(gpus, key=lambda x: x.load)
                    if best_gpu.load < 0.9:
                        config["num_gpu"] = 1
                    else:
                        config["num_gpu"] = 0
                else:
                    config["num_gpu"] = 0
            except:
                config["num_gpu"] = 0
        else:
            config["num_gpu"] = 0
        
        # 内存配置
        if memory_gb is None:
            try:
                memory_gb = psutil.virtual_memory().total / (1024**3)
            except:
                memory_gb = 8  # 默认值
        
        # 根据内存大小调整配置
        if memory_gb >= 32:
            config["keep_alive"] = "30m"
        elif memory_gb >= 16:
            config["keep_alive"] = "15m"
        else:
            config["keep_alive"] = "5m"
        
        return config  


# 测试代码  
if __name__ == "__main__":  
        
    # 测试模型名称，请替换为实际的模型名称  
    model_name = "qwen:7b" # 或其他支持的模型  

    model_path = "models/gpt2"
    
    print("=== Ollama推理优化版测试 ===\n")
    
    # 1. 创建优化配置
    try:
        config = OllamaInference.create_optimal_config(model_name)
        print(f"检测到的优化配置: {config}")
    except Exception as e:
        print(f"创建配置时出错: {e}")
        config = {"model_name": model_name}
    
    # 2. 使用上下文管理器初始化
    try:
        with OllamaInference(**config) as inference:
            
            # 检查模型是否可用
            if not inference.is_model_available():
                print(f"模型 {model_name} 不存在，请先使用 'ollama pull {model_name}' 下载模型")
                sys.exit(1)
            
            # 显示健康状态
            health = inference.get_health_status()
            print(f"推理器健康状态: {json.dumps(health, indent=2, ensure_ascii=False)}")
            
            # 测试普通问答
            print("\n=== 测试普通问答 ===")
            query = "高血压患者应该注意什么？"  
            print("问题:", query)  
            
            answer = inference.answer_question(query)  
            print("回答:", answer)
            
            # 测试性能指标
            metrics = inference.get_performance_metrics()
            print(f"性能指标: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
            
            # 测试RAG问答
            print("\n=== 测试RAG问答 ===")
            context_docs = [  
                "高血压患者应该控制盐的摄入量，每日不超过5克。",  
                "高血压患者应该适当增加体育锻炼，但避免剧烈运动。",  
                "高血压患者应该保持心情舒畅，避免精神紧张。"  
            ]  
            
            rag_answer = inference.answer_with_rag(query, context_docs)  
            print("RAG回答:", rag_answer)  
            
            # 测试流式生成
            print("\n=== 测试流式生成 ===")
            print("问题:", query)
            print("流式回答:", end=" ", flush=True)
            for text in inference.stream_answer(query):  
                print(text, end="", flush=True)
            print()
            
            # 测试嵌入
            print("\n=== 测试嵌入 ===")
            embeddings = inference.embeddings(["高血压是什么？"])  
            if embeddings and embeddings[0]:
                print(f"嵌入向量维度: {len(embeddings[0])}")
                print(f"嵌入向量前5个值: {embeddings[0][:5]}")
            else:
                print("嵌入向量获取失败")
            
            # 最终性能统计
            final_metrics = inference.get_performance_metrics()
            print(f"\n=== 最终性能统计 ===")
            print(f"总请求数: {final_metrics['total_requests']}")
            print(f"成功请求数: {final_metrics['successful_requests']}")
            print(f"失败请求数: {final_metrics['failed_requests']}")
            print(f"成功率: {final_metrics['success_rate']:.2%}")
            print(f"平均延迟: {final_metrics['avg_latency']:.2f}秒")
            print(f"总Token数: {final_metrics['total_tokens_generated']}")
            
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  