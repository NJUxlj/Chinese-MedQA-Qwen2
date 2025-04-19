"""  
完整推理流水线，整合模型推理和RAG检索功能  
"""  
import os  
import sys  
import time  
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator, Callable  
import torch  

# 确保可以导入项目其他模块  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from utils.logger import get_logger  
from config.model_config import ModelConfig  
from config.rag_config import RAGConfig  
from inference.inference_utils import (  
    format_prompt,   
    format_rag_prompt,   
    postprocess_response,   
    measure_latency,  
    get_inference_params  
)  

# 导入推理模块  
from inference.fastllm_inference import FastLLMInference, FASTLLM_AVAILABLE  
from inference.vllm_inference import VLLMInference, VLLM_AVAILABLE  
from inference.api_inference import (  
    ZhipuAIInference,   
    DashScopeInference,   
    OpenAIInference,  
    ZHIPUAI_AVAILABLE,  
    DASHSCOPE_AVAILABLE,  
    OPENAI_AVAILABLE  
)  

logger = get_logger("inference_pipeline")  

class InferencePipeline:  
    """  
    推理流水线，整合模型推理和RAG检索  
    """  
    def __init__(self,   
                inference_type: str,  
                inference_params: Dict,  
                use_rag: bool = True,  
                retriever=None,  
                top_k: int = 3):  
        """  
        初始化推理流水线  
        
        Args:  
            inference_type: 推理类型，支持fastllm、vllm、zhipuai、dashscope、openai  
            inference_params: 推理参数字典  
            use_rag: 是否使用RAG  
            retriever: 检索器对象  
            top_k: 检索的文档数量  
        """  
        self.inference_type = inference_type  
        self.inference_params = inference_params  
        self.use_rag = use_rag  
        self.retriever = retriever  
        self.top_k = top_k  
        
        # 初始化模型  
        self.inference = self._create_inference()  
        
        # 是否支持流式生成  
        self.support_stream = True  
        
        logger.info(f"推理流水线初始化完成，推理类型: {inference_type}, 使用RAG: {use_rag}")  
    
    def _create_inference(self):  
        """  
        创建推理对象  
        
        Returns:  
            推理对象  
        """  
        if self.inference_type == "fastllm":  
            if not FASTLLM_AVAILABLE:  
                raise ImportError("请先安装fastllm库")  
            return FastLLMInference(**self.inference_params)  
        
        elif self.inference_type == "vllm":  
            if not VLLM_AVAILABLE:  
                raise ImportError("请先安装vllm库")  
            return VLLMInference(**self.inference_params)  
        
        elif self.inference_type == "zhipuai":  
            if not ZHIPUAI_AVAILABLE:  
                raise ImportError("请先安装zhipuai库")  
            return ZhipuAIInference(**self.inference_params)  
        
        elif self.inference_type == "dashscope":  
            if not DASHSCOPE_AVAILABLE:  
                raise ImportError("请先安装dashscope库")  
            return DashScopeInference(**self.inference_params)  
        
        elif self.inference_type == "openai":  
            if not OPENAI_AVAILABLE:  
                raise ImportError("请先安装openai库")  
            return OpenAIInference(**self.inference_params)  
        
        else:  
            raise ValueError(f"不支持的推理类型: {self.inference_type}")  
    
    @measure_latency  
    def retrieve_documents(self, query: str) -> List[str]:  
        """  
        检索相关文档  
        
        Args:  
            query: 用户查询  
            
        Returns:  
            相关文档列表  
        """  
        if not self.use_rag or self.retriever is None:  
            return []  
        
        try:  
            # 调用检索器  
            start_time = time.time()  
            docs = self.retriever.retrieve(query, top_k=self.top_k)  
            logger.info(f"检索耗时: {time.time() - start_time:.4f}秒")  
            
            # 提取文档内容  
            return [doc.page_content for doc in docs]  
            
        except Exception as e:  
            logger.error(f"文档检索出错: {str(e)}")  
            return []  
    
    @measure_latency  
    def answer(self,   
              query: str,   
              history: Optional[List[Dict[str, str]]] = None,  
              system_prompt: Optional[str] = None,  
              **kwargs) -> Dict:  
        """  
        回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            
        Returns:  
            包含回答和相关信息的字典  
        """  
        start_time = time.time()  
        
        # 检索相关文档  
        context_docs = []  
        if self.use_rag and self.retriever is not None:  
            context_docs = self.retrieve_documents(query)  
            
        # 生成回答  
        answer = ""  
        if context_docs:  
            # 使用RAG进行回答  
            answer = self.inference.answer_with_rag(  
                query=query,  
                context_docs=context_docs,  
                history=history,  
                system_prompt=system_prompt,  
                **kwargs  
            )  
        else:  
            # 直接回答  
            answer = self.inference.answer_question(  
                query=query,  
                history=history,  
                system_prompt=system_prompt,  
                **kwargs  
            )  
        
        end_time = time.time()  
        
        # 返回结果  
        return {  
            "query": query,  
            "answer": answer,  
            "context_docs": context_docs,  
            "history": history,  
            "latency": end_time - start_time  
        }  
    
    def stream_answer(self,   
                     query: str,   
                     history: Optional[List[Dict[str, str]]] = None,  
                     system_prompt: Optional[str] = None,  
                     callback: Optional[Callable[[str], None]] = None,  
                     **kwargs) -> Iterator[Dict]:  
        """  
        流式回答问题  
        
        Args:  
            query: 用户查询  
            history: 历史对话记录  
            system_prompt: 系统提示词  
            callback: 回调函数，接收生成的文本片段  
            
        Yields:  
            包含回答片段和相关信息的字典  
        """  
        start_time = time.time()  
        
        # 检索相关文档  
        context_docs = []  
        if self.use_rag and self.retriever is not None:  
            context_docs = self.retrieve_documents(query)  
        
        # 生成回答  
        answer_chunks = []  
        
        try:  
            if hasattr(self.inference, "stream_answer"):  
                # API模型使用stream_answer方法  
                if context_docs:  
                    # 使用RAG进行回答（API模型不支持流式的answer_with_rag，所以先构造提示词）  
                    rag_system_prompt = system_prompt  
                    if system_prompt is None:  
                        rag_system_prompt = "你是一个专业的医疗助手，请根据用户的问题和提供的参考信息，给出准确、专业的医疗建议。请注意，你的建议仅供参考，不能替代专业医生的诊断和治疗。"  
                    
                    # 整合上下文信息  
                    context_text = "\n\n".join([f"文档{i+1}：{doc}" for i, doc in enumerate(context_docs)])  
                    
                    # 创建包含上下文的用户查询  
                    context_query = f"问题：{query}\n\n参考信息：\n{context_text}\n\n请根据以上参考信息回答问题。"  
                    
                    # 流式生成  
                    for chunk in self.inference.stream_answer(context_query, history, rag_system_prompt, **kwargs):  
                        answer_chunks.append(chunk)  
                        if callback:  
                            callback(chunk)  
                        yield {  
                            "query": query,  
                            "answer_chunk": chunk,  
                            "answer": "".join(answer_chunks),  
                            "context_docs": context_docs,  
                            "finished": False,  
                            "latency": time.time() - start_time  
                        }  
                else:  
                    # 直接回答  
                    for chunk in self.inference.stream_answer(query, history, system_prompt, **kwargs):  
                        answer_chunks.append(chunk)  
                        if callback:  
                            callback(chunk)  
                        yield {  
                            "query": query,  
                            "answer_chunk": chunk,  
                            "answer": "".join(answer_chunks),  
                            "context_docs": context_docs,  
                            "finished": False,  
                            "latency": time.time() - start_time  
                        }  
            else:  
                # 本地模型使用stream_generate方法  
                if context_docs:  
                    # 构造RAG提示词  
                    prompt = format_rag_prompt(query, context_docs, history, system_prompt)  
                    
                    # 流式生成  
                    for chunk in self.inference.stream_generate(prompt, **kwargs):  
                        answer_chunks.append(chunk)  
                        if callback:  
                            callback(chunk)  
                        yield {  
                            "query": query,  
                            "answer_chunk": chunk,  
                            "answer": "".join(answer_chunks),  
                            "context_docs": context_docs,  
                            "finished": False,  
                            "latency": time.time() - start_time  
                        }  
                else:  
                    # 构造普通提示词  
                    prompt = format_prompt(query, history, system_prompt)  
                    
                    # 流式生成  
                    for chunk in self.inference.stream_generate(prompt, **kwargs):  
                        answer_chunks.append(chunk)  
                        if callback:  
                            callback(chunk)  
                        yield {  
                            "query": query,  
                            "answer_chunk": chunk,  
                            "answer": "".join(answer_chunks),  
                            "context_docs": context_docs,  
                            "finished": False,  
                            "latency": time.time() - start_time  
                        }  
        
        except Exception as e:  
            logger.error(f"流式生成出错: {str(e)}")  
            error_chunk = f"生成出错: {str(e)}"  
            answer_chunks.append(error_chunk)  
            if callback:  
                callback(error_chunk)  
            yield {  
                "query": query,  
                "answer_chunk": error_chunk,  
                "answer": "".join(answer_chunks),  
                "context_docs": context_docs,  
                "finished": False,  
                "latency": time.time() - start_time  
            }  
        
        # 最后一个chunk标记为完成  
        yield {  
            "query": query,  
            "answer_chunk": "",  
            "answer": "".join(answer_chunks),  
            "context_docs": context_docs,  
            "finished": True,  
            "latency": time.time() - start_time  
        }  
    
    def update_retriever(self, retriever):  
        """  
        更新检索器  
        
        Args:  
            retriever: 新的检索器对象  
        """  
        self.retriever = retriever  
        logger.info("检索器已更新")  
    
    def update_inference(self, inference_type: str, inference_params: Dict):  
        """  
        更新推理模型  
        
        Args:  
            inference_type: 推理类型  
            inference_params: 推理参数  
        """  
        self.inference_type = inference_type  
        self.inference_params = inference_params  
        self.inference = self._create_inference()  
        logger.info(f"推理模型已更新，类型: {inference_type}")  


# 测试代码  
if __name__ == "__main__":  
    # 测试需要提供有效的检索器和模型参数  
    
    # 模拟一个简单的检索器  
    class DummyRetriever:  
        def retrieve(self, query, top_k=3):  
            # 模拟的文档类  
            class Doc:  
                def __init__(self, content):  
                    self.page_content = content  
            
            # 根据查询返回一些假文档  
            if "高血压" in query:  
                return [  
                    Doc("高血压患者应该控制盐的摄入量，每日不超过5克。"),  
                    Doc("高血压患者应该适当增加体育锻炼，但避免剧烈运动。"),  
                    Doc("高血压患者应该保持心情舒畅，避免精神紧张。")  
                ][:top_k]  
            elif "糖尿病" in query:  
                return [  
                    Doc("糖尿病患者应该控制碳水化合物的摄入。"),  
                    Doc("糖尿病患者应该定期监测血糖水平。"),  
                    Doc("糖尿病患者应该保持适当的体重。")  
                ][:top_k]  
            else:  
                return []  
    
    # 创建推理流水线  
    try:  
        # 可以选择不同的推理类型进行测试  
        if VLLM_AVAILABLE:  
            pipeline = InferencePipeline(  
                inference_type="vllm",  
                inference_params={  
                    "model_path": "/path/to/your/model",  # 替换为实际的模型路径  
                    "tensor_parallel_size": 1  
                },  
                use_rag=True,  
                retriever=DummyRetriever(),  
                top_k=2  
            )  
            
            # 测试普通问答  
            query = "高血压患者应该注意什么？"  
            print("问题:", query)  
            
            result = pipeline.answer(query)  
            print("回答:", result["answer"])  
            print("相关文档:")  
            for i, doc in enumerate(result["context_docs"]):  
                print(f"  {i+1}. {doc}")  
            print(f"延迟: {result['latency']:.4f}秒")  
            
            # 测试流式生成  
            print("\n流式生成:")  
            full_answer = ""  
            for chunk in pipeline.stream_answer(query):  
                print(chunk["answer_chunk"], end="", flush=True)  
                if chunk["finished"]:  
                    full_answer = chunk["answer"]  
                    print(f"\n延迟: {chunk['latency']:.4f}秒")  
            
            print(f"完整回答: {full_answer}")  
            
        elif ZHIPUAI_AVAILABLE:  
            # 智谱API测试  
            api_key = "your_zhipuai_api_key"  # 替换为实际的API密钥  
            pipeline = InferencePipeline(  
                inference_type="zhipuai",  
                inference_params={  
                    "api_key": api_key,  
                    "model": "glm-4"  
                },  
                use_rag=True,  
                retriever=DummyRetriever(),  
                top_k=2  
            )  
            
            # 测试普通问答  
            query = "糖尿病患者应该注意什么？"  
            print("问题:", query)  
            
            result = pipeline.answer(query)  
            print("回答:", result["answer"])  
            print("相关文档:")  
            for i, doc in enumerate(result["context_docs"]):  
                print(f"  {i+1}. {doc}")  
            print(f"延迟: {result['latency']:.4f}秒")  
            
            # 测试流式生成  
            print("\n流式生成:")  
            full_answer = ""  
            for chunk in pipeline.stream_answer(query):  
                print(chunk["answer_chunk"], end="", flush=True)  
                if chunk["finished"]:  
                    full_answer = chunk["answer"]  
                    print(f"\n延迟: {chunk['latency']:.4f}秒")  
            
            print(f"完整回答: {full_answer}")  
            
        else:  
            print("请安装vLLM或zhipuai等库以进行测试")  
            
    except Exception as e:  
        print(f"测试出错: {str(e)}")  