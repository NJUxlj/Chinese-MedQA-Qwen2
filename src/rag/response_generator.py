# src/rag/response_generator.py  

from typing import Dict, List, Optional, Union, Any, Tuple  
import re  
import time  
from models.base_model import BaseGenerativeModel  
from utils.logger import setup_logger  

logger = setup_logger(__name__)  

class ResponseGenerator:  
    """  
    响应生成器，负责生成RAG增强的最终回答  
    """  
    
    def __init__(  
        self,  
        model: BaseGenerativeModel,  
        max_new_tokens: int = 1024,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        template: Optional[str] = None,  
        use_chatml_format: bool = True  
    ):  
        """  
        初始化响应生成器  
        
        Args:  
            model: 生成模型  
            max_new_tokens: 生成的最大token数量  
            temperature: 生成温度  
            top_p: 生成top_p值  
            template: 响应模板  
            use_chatml_format: 是否使用ChatML格式  
        """  
        self.model = model  
        self.max_new_tokens = max_new_tokens  
        self.temperature = temperature  
        self.top_p = top_p  
        self.use_chatml_format = use_chatml_format  
        
        # 设置响应模板  
        if template:  
            self.template = template  
        else:  
            if use_chatml_format:  
                # ChatML格式模板  
                self.template = (  
                    "<|im_start|>system\n"  
                    "你是一个专业的医疗助手，请根据提供的医疗信息，准确回答用户的问题。"  
                    "如果提供的信息不足以回答问题，请基于你的医学知识给出合理回答，但要明确指出哪些是你的补充解释。"  
                    "<|im_end|>\n"  
                    "<|im_start|>user\n"  
                    "根据以下信息回答我的问题：\n\n"  
                    "{context}\n\n"  
                    "问题: {query}"  
                    "<|im_end|>\n"  
                    "<|im_start|>assistant\n"  
                )  
            else:  
                # 普通格式模板  
                self.template = (  
                    "请根据以下信息回答问题。如果无法从提供的信息中找到答案，请基于可靠的医学知识回答，"  
                    "并注明这是基于一般医学知识的回答。\n\n"  
                    "相关信息：\n{context}\n\n"  
                    "问题：{query}\n\n"  
                    "回答："  
                )  
    
    def format_prompt(self, query: str, context: str) -> str:  
        """  
        使用模板格式化提示  
        
        Args:  
            query: 查询文本  
            context: 上下文  
            
        Returns:  
            格式化后的提示  
        """  
        return self.template.format(query=query, context=context)  
    
    
    
    
    def process_generated_response(self, response: str) -> str:
        """
        处理生成的响应文本
        
        Args:
            response: 生成的原始响应
            
        Returns:
            处理后的响应文本
        """
        # 移除可能的前缀
        if response.startswith("回答："):
            response = response[3:]
        
        # 处理ChatML格式输出
        if self.use_chatml_format:
            # 尝试提取assistant部分
            match = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", response, re.DOTALL)
            if match:
                response = match.group(1).strip()
            else:
                # 尝试移除结束标记后的内容
                response = re.sub(r"<\|im_end\|>.*$", "", response, flags=re.DOTALL).strip()
        
        # 移除多余的空白行
        response = re.sub(r"\n{3,}", "\n\n", response)
        
        # 如果响应为空，返回默认消息
        if not response.strip():
            return "很抱歉，我无法针对这个问题生成有效回答。请尝试重新表述您的问题。"
        
        return response.strip()
    
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成增强响应
        
        Args:
            prompt: 包含查询、上下文和格式化提示的字典
            
        Returns:
            包含响应和元数据的字典
        """
        formatted_prompt = prompt.get("formatted_prompt")
        query = prompt.get("query", "")
        context = prompt.get("context", "")
        
        # 如果没有提供格式化提示，则创建一个
        if not formatted_prompt:
            formatted_prompt = self.format_prompt(query, context)
        
        start_time = time.time()
        
        try:
            # 使用模型生成回答
            raw_response = self.model.generate(
                formatted_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # 处理生成的响应
            processed_response = self.process_generated_response(raw_response)
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            result = {
                "query": query,
                "context": context,
                "prompt": formatted_prompt,
                "raw_response": raw_response,
                "response": processed_response,
                "source_documents": prompt.get("documents", []),
                "metadata": {
                    "generation_time": generation_time,
                    "model": getattr(self.model, "model_name", "unknown"),
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"生成响应时出错: {e}")
            error_message = f"生成回答时发生错误: {str(e)}"
            
            return {
                "query": query,
                "context": context,
                "prompt": formatted_prompt,
                "response": error_message,
                "error": str(e),
                "source_documents": prompt.get("documents", [])
            }
    
    def generate_with_streaming(self, prompt: Dict[str, Any], callback=None) -> Dict[str, Any]:
        """
        使用流式输出生成增强响应
        
        Args:
            prompt: 包含查询、上下文和格式化提示的字典
            callback: 流式输出回调函数，接收部分响应文本
            
        Returns:
            包含响应和元数据的字典
        """
        formatted_prompt = prompt.get("formatted_prompt")
        query = prompt.get("query", "")
        context = prompt.get("context", "")
        
        # 如果没有提供格式化提示，则创建一个
        if not formatted_prompt:
            formatted_prompt = self.format_prompt(query, context)
        
        start_time = time.time()
        
        # 检查模型是否支持流式输出
        if not hasattr(self.model, "generate_streaming"):
            logger.warning("模型不支持流式输出，回退到标准生成")
            return self.generate(prompt)
        
        try:
            # 初始化响应文本
            full_response = ""
            
            # 使用模型的流式生成方法
            for response_chunk in self.model.generate_streaming(
                formatted_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            ):
                # 更新完整响应
                full_response += response_chunk
                
                # 如果提供了回调函数，调用它
                if callback:
                    callback(response_chunk)
            
            # 处理生成的响应
            processed_response = self.process_generated_response(full_response)
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            result = {
                "query": query,
                "context": context,
                "prompt": formatted_prompt,
                "raw_response": full_response,
                "response": processed_response,
                "source_documents": prompt.get("documents", []),
                "metadata": {
                    "generation_time": generation_time,
                    "model": getattr(self.model, "model_name", "unknown"),
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                    "streaming": True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"流式生成响应时出错: {e}")
            error_message = f"生成回答时发生错误: {str(e)}"
            
            return {
                "query": query,
                "context": context,
                "prompt": formatted_prompt,
                "response": error_message,
                "error": str(e),
                "source_documents": prompt.get("documents", [])
            }
    
    def evaluate_response_quality(self, response: str, context: str, query: str) -> Dict[str, float]:
        """
        评估生成响应的质量
        
        Args:
            response: 生成的响应
            context: 上下文
            query: 查询
            
        Returns:
            质量评分字典
        """
        # 一个简单的质量评估实现
        scores = {}
        
        # 响应长度评分
        response_length = len(response)
        if response_length < 50:
            length_score = 0.5
        elif response_length < 100:
            length_score = 0.7
        elif response_length < 300:
            length_score = 1.0
        else:
            length_score = 0.9  # 过长可能不够精炼
        
        scores["length_score"] = length_score
        
        # 上下文利用评分
        # 检查响应中是否包含上下文中的关键信息
        context_words = set(re.findall(r'\w+', context.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        # 计算交集比例
        if context_words:
            context_score = len(context_words.intersection(response_words)) / min(len(context_words), 100)
            context_score = min(context_score * 2, 1.0)  # 归一化，最高1.0
        else:
            context_score = 0.5
        
        scores["context_score"] = context_score
        
        # 查询相关性评分
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # 检查响应中是否提及查询中的关键词
        if query_words:
            query_score = len(query_words.intersection(response_words)) / len(query_words)
            query_score = min(query_score * 1.5, 1.0)  # 归一化，最高1.0
        else:
            query_score = 0.7
        
        scores["query_score"] = query_score
        
        # 计算总体质量分数
        overall_score = (length_score * 0.2 + context_score * 0.4 + query_score * 0.4)
        scores["overall_score"] = overall_score
        
        return scores
    
    def handle_citations(self, response: str, documents: List[Dict[str, Any]]) -> str:
        """
        处理响应中的引用，将文档索引转换为实际引用
        
        Args:
            response: 生成的响应
            documents: 源文档列表
            
        Returns:
            处理引用后的响应
        """
        # 如果没有文档或响应中没有引用，直接返回
        if not documents or "[文档" not in response:
            return response
        
        # 查找响应中的文档引用
        citation_pattern = r'\[文档\s*(\d+)(?:-\d+)?\]'
        matches = re.finditer(citation_pattern, response)
        
        # 创建引用映射
        citations = {}
        for match in matches:
            doc_idx = int(match.group(1)) - 1  # 转为0索引
            if 0 <= doc_idx < len(documents):
                doc = documents[doc_idx]
                source = doc.get("metadata", {}).get("source", f"来源 {doc_idx+1}")
                citations[match.group(0)] = f"[{source}]"
        
        # 替换引用
        processed_response = response
        for old_citation, new_citation in citations.items():
            processed_response = processed_response.replace(old_citation, new_citation)
        
        return processed_response
    
    def format_response_with_metadata(self, result: Dict[str, Any], include_sources: bool = True) -> str:
        """
        将响应结果格式化为包含元数据的文本
        
        Args:
            result: 生成结果字典
            include_sources: 是否包含源文档信息
            
        Returns:
            格式化后的响应文本
        """
        response = result.get("response", "")
        source_documents = result.get("source_documents", [])
        
        formatted_response = response
        
        # 如果需要包含源文档信息
        if include_sources and source_documents:
            formatted_response += "\n\n参考资料："
            for i, doc in enumerate(source_documents[:3]):  # 限制显示前3个文档
                source = doc.get("metadata", {}).get("source", f"来源 {i+1}")
                score_info = ""
                if "score" in doc:
                    score_info = f" (相关度: {doc['score']:.2f})"
                elif "rerank_score" in doc:
                    score_info = f" (相关度: {doc['rerank_score']:.2f})"
                
                formatted_response += f"\n[{i+1}] {source}{score_info}"
                # 添加源文档的简短摘要
                text = doc.get("text", "")
                if text:
                    summary = text[:100] + "..." if len(text) > 100 else text
                    formatted_response += f"\n    {summary}"
        
        return formatted_response