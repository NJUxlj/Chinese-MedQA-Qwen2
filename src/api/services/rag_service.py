"""
RAG服务
负责知识检索和上下文增强
"""

from typing import Dict, Any, List, Optional
import sys
import threading
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings
from providers.llm_provider import LLMProvider
from providers.embedding_provider import EmbeddingProvider
from providers.reranker_provider import RerankerProvider
from rag.rag_pipeline import RAGPipeline
from utils.logger import setup_logger

class RAGService:
    """RAG服务类, 专门用于为 rag http api 提供服务"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        reranker_provider: Optional[RerankerProvider] = None):
        """初始化RAG服务"""
        self._pipelines_lock = threading.RLock()
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.logger = setup_logger(name=self.__class__.__name__, level="INFO")

        self.initialized = False
        self.rag_pipeline = None

        self.initialize()

    def initialize(self) -> None:
        """初始化RAG服务"""
        if self.initialized:
            return
        self.logger.info("正在初始化RAG服务 ...")
        self.rag_pipeline = RAGPipeline(
            llm_provider=self.llm_provider,
            embedding_provider=self.embedding_provider,
            reranker_provider=self.reranker_provider)
        self.initialized = True
        self.logger.info("RAG服务初始化成功, 模型: %s", self.llm_provider.model_name)



    def generate_response(self, query: str,
                          model_name: Optional[str] = None,
                          top_k: int = 5) -> Dict[str, Any]:
        """
        生成RAG增强响应

        Args:
            query: 查询文本
            model_name: 模型名称
            top_k: 检索文档数量

        Returns:
            响应结果
        """
        if not self.rag_pipeline:
            raise ValueError("RAGPipeline 未初始化")

        result = self.rag_pipeline.generate_response(query_text=query, top_k=top_k)

        return {
            "response": result.get("response", ""),
            "context": result.get("context", ""),
            "source_documents": result.get("source_documents", []),
        }




# 单例模式
_rag_service = None
_lock = threading.Lock()

def get_rag_service(
    llm_provider: LLMProvider = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    reranker_provider: Optional[RerankerProvider] = None) -> RAGService:
    """获取RAG服务单例

    Args:
        llm_provider: LLM提供者，如果单例未初始化则必须提供
    """
    global _rag_service
    if _rag_service is None:
        if llm_provider is None or embedding_provider is None or reranker_provider is None:
            raise ValueError("首次调用 get_rag_service 必须提供 llm_provider, embedding_provider, reranker_provider")
        with _lock:
            if _rag_service is None:
                _rag_service = RAGService(
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    reranker_provider=reranker_provider)    
                    
    return _rag_service
