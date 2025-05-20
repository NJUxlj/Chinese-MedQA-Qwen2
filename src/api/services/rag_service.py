"""
RAG服务
负责知识检索和上下文增强
"""

from typing import Dict, Any, List, Optional, Union
import os
import threading
import logging
from pathlib import Path

# 导入RAG组件
from rag.rag_pipeline import RAGPipeline
from knowledge_base.kb_manager import KnowledgeBaseManager
from services.model_service import get_model_service, ModelService
from services.embedding_service import get_embedding_service, EmbeddingService

logger = logging.getLogger(__name__)

class RAGService:
    """RAG服务类"""
    
    def __init__(self):
        """初始化RAG服务"""
        self.pipelines: Dict[str, RAGPipeline] = {}
        self._pipelines_lock = threading.RLock()
        
        # 获取服务依赖
        self.model_service = get_model_service()
        self.embedding_service = get_embedding_service()
        
        # 初始化知识库管理器
        self.kb_manager = KnowledgeBaseManager(
            index_dir=os.environ.get("KB_INDEX_DIR", "knowledge_base/indices")
        )
        
        # 默认检索器类型
        self.default_retriever_type = os.environ.get("DEFAULT_RETRIEVER", "hybrid")
        
        # 配置初始化标志
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化RAG服务"""
        if self.initialized:
            return
        
        logger.info("初始化RAG服务")
        
        try:
            # 初始化数据库管理器
            self.kb_manager.initialize()
            
            # 如果环境变量设置了默认知识库，则创建默认管道
            default_kb = os.environ.get("DEFAULT_KB", None)
            if default_kb:
                self.get_or_create_pipeline(default_kb)
            
            self.initialized = True
            logger.info("RAG服务初始化成功")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise
    
    def get_or_create_pipeline(self, kb_name: str) -> RAGPipeline:
        """
        获取或创建RAG管道
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            RAG管道实例
        """
        with self._pipelines_lock:
            # 检查是否已存在
            if kb_name in self.pipelines:
                return self.pipelines[kb_name]
            
            logger.info(f"创建RAG管道: {kb_name}")
            
            try:
                # 获取知识库索引路径
                index_path = self.kb_manager.get_index_path(kb_name)
                
                if not index_path:
                    raise ValueError(f"知识库 {kb_name} 不存在或索引未创建")
                
                # 获取嵌入模型
                embedding_manager = self.embedding_service.get_default_embedding_manager()
                
                # 创建RAG管道
                pipeline = RAGPipeline(
                    retriever_type=self.default_retriever_type,
                    embedding_model_name=embedding_manager.model_name,
                    embedding_dimension=embedding_manager.embedding_dimension,
                    index_path=str(index_path),
                    top_k=5,
                    model=None,  # 模型将按需加载
                    use_reranker=True,
                    hybrid_weight=0.7
                )
                
                # 保存管道
                self.pipelines[kb_name] = pipeline
                
                return pipeline
                
            except Exception as e:
                logger.error(f"创建RAG管道 {kb_name} 失败: {e}")
                raise
    
    def retrieve(self, kb_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行知识检索
        
        Args:
            kb_name: 知识库名称
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        # 获取RAG管道
        pipeline = self.get_or_create_pipeline(kb_name)
        
        # 执行检索
        documents = pipeline.query(query, top_k)
        
        return documents
    
    def build_context(self, kb_name: str, query: str, top_k: int = 5) -> str:
        """
        构建增强上下文
        
        Args:
            kb_name: 知识库名称
            query: 查询文本
            top_k: 检索文档数量
            
        Returns:
            构建的上下文
        """
        # 获取RAG管道
        pipeline = self.get_or_create_pipeline(kb_name)
        
        # 构建上下文
        context = pipeline.build_context(query, top_k)
        
        return context
    
    def generate_response(self, kb_name: str, query: str, 
                          model_name: Optional[str] = None,
                          top_k: int = 5) -> Dict[str, Any]:
        """
        生成RAG增强响应
        
        Args:
            kb_name: 知识库名称
            query: 查询文本
            model_name: 模型名称
            top_k: 检索文档数量
            
        Returns:
            响应结果
        """
        # 获取RAG管道
        pipeline = self.get_or_create_pipeline(kb_name)
        
        # 获取模型
        model = self.model_service.get_model(model_name)
        
        # 设置模型
        pipeline.set_model(model)
        
        # 生成响应
        response = pipeline.generate_response(query, top_k)
        
        return response
    
    def get_available_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        获取可用知识库列表
        
        Returns:
            知识库信息列表
        """
        return self.kb_manager.list_knowledge_bases()
    
    def create_knowledge_base(self, kb_name: str, description: str) -> bool:
        """
        创建新知识库
        
        Args:
            kb_name: 知识库名称
            description: 知识库描述
            
        Returns:
            是否创建成功
        """
        return self.kb_manager.create_knowledge_base(kb_name, description)
    
    def delete_knowledge_base(self, kb_name: str) -> bool:
        """
        删除知识库
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            是否删除成功
        """
        # 如果有对应的管道，先移除
        with self._pipelines_lock:
            if kb_name in self.pipelines:
                del self.pipelines[kb_name]
        
        return self.kb_manager.delete_knowledge_base(kb_name)
    
    def add_documents(self, kb_name: str, documents: List[Dict[str, Any]]) -> bool:
        """
        向知识库添加文档
        
        Args:
            kb_name: 知识库名称
            documents: 文档列表
            
        Returns:
            是否添加成功
        """
        success = self.kb_manager.add_documents(kb_name, documents)
        
        # 如果添加成功且有对应的管道，更新索引
        if success and kb_name in self.pipelines:
            try:
                index_path = self.kb_manager.get_index_path(kb_name)
                pipeline = self.pipelines[kb_name]
                pipeline.update_retriever_index(documents, str(index_path))
            except Exception as e:
                logger.error(f"更新RAG索引失败: {e}")
                return False
        
        return success

# 单例模式
_rag_service = None
_lock = threading.Lock()

def get_rag_service() -> RAGService:
    """
    获取RAG服务单例
    
    Returns:
        RAG服务实例
    """
    global _rag_service
    
    if _rag_service is None:
        with _lock:
            if _rag_service is None:
                _rag_service = RAGService()
                # 初始化服务
                _rag_service.initialize()
    
    return _rag_service