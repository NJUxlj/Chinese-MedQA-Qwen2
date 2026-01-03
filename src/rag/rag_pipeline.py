# src/rag/rag_pipeline.py

import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from langchain_core.documents import Document

from typing import Dict, List, Optional, Union, Any, Tuple
from knowledge_base.retrieval.knn_retriever import KNNRetriever
from knowledge_base.retrieval.similarity_retriever import SimilarityRetriever
from knowledge_base.retrieval.bm25_retriever import BM25Retriever
from knowledge_base.retrieval.l2_retriever import L2Retriever
from knowledge_base.embedding.embedding_manager import EmbeddingManager
from rag.query_processor import QueryProcessor
from rag.context_builder import ContextBuilder
from rag.response_generator import ResponseGenerator
from models.base_model import BaseGenerativeModel
from utils.logger import setup_logger

from config.rag_config import RAGConfig
from config.embedding_config import EmbeddingConfig

logger = setup_logger(__name__)

class RAGPipeline:
    """
    RAG流水线，整合查询处理、文档检索、上下文构建和响应生成
    """
    
    def __init__(
        self,
        config: RAGConfig
    ):
        """
        初始化RAG流水线
        
        Args:
            retriever_type: 检索器类型
            embedding_model_name: 嵌入模型名称
            embedding_dimension: 嵌入维度
            index_path: 索引路径
            top_k: 检索的文档数量
            model: 用于生成响应的模型
            cache_dir: 缓存目录
            reranker_model_name: 重排序模型名称
            use_reranker: 是否使用重排序
            hybrid_weight: 混合检索时的权重
            chunk_size: 上下文构建时的块大小
            chunk_overlap: 上下文块重叠大小
            response_template: 响应模板
            max_new_tokens: 生成的最大token数量
            temperature: 生成温度
            top_p: 生成top_p值
        """
        self.config = config


        self.retriever_type: str = self.config.retriever_type
        self.embedding_model_name: str = self.config.embedding_model_name
        self.embedding_dimension: int = self.config.embedding_dimension
        self.index_path: Optional[str] = self.config.index_path
        self.top_k: int = self.config.top_k
        self.model: Optional[BaseGenerativeModel] = self.config.model
        self.cache_dir: Optional[str] = self.config.cache_dir
        self.reranker_model_provider: str = self.config.reranker_model_provider
        self.reranker_model_path: Optional[str] = self.config.reranker_model_path
        self.reranker_model_name: Optional[str] = self.config.reranker_model_name
        self.use_reranker: bool = self.config.use_reranker
        self.hybrid_weight: float = self.config.hybrid_weight
        self.chunk_size: int = self.config.chunk_size
        self.chunk_overlap: int = self.config.chunk_overlap
        self.response_template: Optional[str] = self.config.response_template
        self.max_new_tokens: int = self.config.max_new_tokens
        self.temperature: float = self.config.temperature
        self.top_p: float = self.config.top_p

    
        
        # 初始化查询处理器
        self.query_processor = QueryProcessor()
        
        # 初始化嵌入管理器 (对于某些检索器需要)
        if self.retriever_type in ["knn", "similarity", "l2", "hybrid"]:
            self.embedding_manager = EmbeddingManager(
                embedding_model_name=self.embedding_model_name,
                cache_dir=self.cache_dir
            )
        else:
            self.embedding_manager = None
        
        # 根据类型初始化检索器
        if self.retriever_type == "knn":
            from config.retriever_config import KNNRetrieverConfig
            config = KNNRetrieverConfig()
            self.retriever = KNNRetriever(
                config=config,
                embedding_manager=self.embedding_manager
            )
        elif self.retriever_type == "similarity":
            from config.retriever_config import SimilarityRetrieverConfig
            config = SimilarityRetrieverConfig()
            self.retriever = SimilarityRetriever(
                config=config,
                embedding_manager=self.embedding_manager
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever()
        elif self.retriever_type == "l2":
            from config.retriever_config import L2RetrieverConfig
            config = L2RetrieverConfig()
            self.retriever = L2Retriever(
                config=config,
                embedding_manager=self.embedding_manager
            )
        elif self.retriever_type == "hybrid":
            from config.retriever_config import KNNRetrieverConfig
            dense_config = KNNRetrieverConfig()
            self.dense_retriever = KNNRetriever(
                config=dense_config,
                embedding_manager=self.embedding_manager
            )
            self.sparse_retriever = BM25Retriever()
            self.retriever = None
        else:
            raise ValueError(f"不支持的检索器类型: {self.retriever_type}")
        
        # 初始化上下文构建器
        self.context_builder = ContextBuilder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            reranker_model_provider=self.reranker_model_provider,
            reranker_model_path=self.reranker_model_path if self.use_reranker else None,
            reranker_model_name=self.reranker_model_name if self.use_reranker else None
        )
        
        # 如果提供了模型，初始化响应生成器
        if self.model:
            self.response_generator = ResponseGenerator(
                model=self.model,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                template=self.response_template
            )
        else:
            self.response_generator = None
    
    def _get_variant_index_path(self, index_path: str, variant: str) -> str:
        """
        获取变体索引路径
        
        Args:
            index_path: 原始索引路径
            variant: 变体类型 ("dense" 或 "sparse")
            
        Returns:
            变体索引路径
        """
        if variant not in ["dense", "sparse"]:
            raise ValueError(f"不支持的变体类型: {variant}")
        
        base_path = Path(index_path)
        if index_path.endswith(".bin"):
            new_name = base_path.stem.replace("_hybrid", "") + f"_{variant}.bin"
        else:
            new_name = base_path.stem + f"_{variant}.bin"
        
        return str(base_path.parent / new_name)
    
    def set_model(self, model: BaseGenerativeModel) -> None:
        """
        设置用于生成响应的模型
        
        Args:
            model: 模型实例
        """
        self.model = model
        self.response_generator = ResponseGenerator(
            model=model,
            max_new_tokens=self.response_generator.max_new_tokens if self.response_generator else 1024,
            temperature=self.response_generator.temperature if self.response_generator else 0.7,
            top_p=self.response_generator.top_p if self.response_generator else 0.9,
            template=self.response_generator.template if self.response_generator else None
        )
    
    def _query_hybrid(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        使用混合检索方式进行查询
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            检索到的文档列表
        """
        k = top_k or self.top_k
        
        dense_results = self.dense_retriever.search(query_text, k*2)
        sparse_results = self.sparse_retriever.search(query_text, k*2)
        
        dense_docs = [(doc.page_content, doc.metadata, score) for doc, score in dense_results]
        sparse_docs = [(doc.page_content, doc.metadata, score) for doc, score in sparse_results]
        
        combined_docs = {}
        
        for i, (content, metadata, score) in enumerate(dense_docs):
            doc_id = metadata.get("doc_id", f"dense_{i}")
            if doc_id not in combined_docs:
                combined_docs[doc_id] = {
                    "text": content,
                    "metadata": metadata,
                    "combined_score": 0
                }
            combined_docs[doc_id]["dense_score"] = score
            combined_docs[doc_id]["combined_score"] += self.hybrid_weight * score
        
        for i, (content, metadata, score) in enumerate(sparse_docs):
            doc_id = metadata.get("doc_id", f"sparse_{i}")
            if doc_id not in combined_docs:
                combined_docs[doc_id] = {
                    "text": content,
                    "metadata": metadata,
                    "combined_score": 0,
                    "dense_score": 0
                }
            combined_docs[doc_id]["sparse_score"] = score
            combined_docs[doc_id]["combined_score"] += (1 - self.hybrid_weight) * score
        
        sorted_docs = sorted(
            combined_docs.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_docs[:k]
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关文档
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            检索到的文档列表
        """
        if not query_text or not query_text.strip():
            logger.warning("查询文本为空")
            return []
        
        processed_query = self.query_processor.process_query(query_text)
        
        if not processed_query or not processed_query.strip():
            logger.warning(f"处理后的查询为空，使用原始查询: {query_text}")
            processed_query = query_text
        
        if not processed_query or not processed_query.strip():
            logger.warning("查询文本为空，无法执行检索")
            return []
        
        k = top_k or self.top_k
        
        if self.retriever_type == "hybrid":
            return self._query_hybrid(processed_query, k)
        
        results = self.retriever.search(processed_query, k)
        
        docs = []
        for doc, score in results:
            docs.append({
                "id": doc.metadata.get("doc_id", "unknown"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return docs
    
    def build_context(self, query_text: str, top_k: Optional[int] = None, 
                     use_reranker: Optional[bool] = None) -> str:
        """
        构建增强上下文
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            构建的上下文字符串
        """
        # 检索相关文档
        retrieved_docs = self.query(query_text, top_k)
        
        if not retrieved_docs:
            logger.warning("没有检索到相关文档")
            return ""
        
        # 确定是否使用重排序
        _use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        
        # 构建上下文
        context = self.context_builder.build_context(
            query=query_text,
            documents=retrieved_docs,
            use_reranker=_use_reranker
        )
        
        return context
    
    def build_rag_prompt(self, query_text: str, top_k: Optional[int] = None,
                       use_reranker: Optional[bool] = None) -> Dict[str, Any]:
        """
        构建RAG提示
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            包含查询、上下文和格式化提示的字典
        """
        # 获取上下文
        context = self.build_context(query_text, top_k, use_reranker)
        
        # 检索相关文档
        documents = self.query(query_text, top_k)
        
        # 构建提示
        prompt = {
            "query": query_text,
            "context": context,
            "documents": documents,
            "formatted_prompt": self.context_builder.format_prompt(query_text, context)
        }
        
        return prompt
    
    def generate_response(self, query_text: str, top_k: Optional[int] = None,
                        use_reranker: Optional[bool] = None) -> Dict[str, Any]:
        """
        生成RAG增强的响应
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            包含响应和元数据的字典
        """
        if not self.model or not self.response_generator:
            raise ValueError("未设置模型，无法生成响应")
        
        # 构建RAG提示
        rag_prompt = self.build_rag_prompt(query_text, top_k, use_reranker)
        
        # 生成响应
        response = self.response_generator.generate(rag_prompt)
        
        return response
    
    def update_retriever_index(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """
        更新检索器索引
        
        Args:
            documents: 文档列表
            save_path: 保存路径
        """
        
        doc_objects = [
            Document(page_content=doc.get("text", doc.get("content", "")), metadata=doc.get("metadata", {}))
            for doc in documents
        ]
        
        if self.retriever_type == "hybrid":
            self.dense_retriever.add_documents(doc_objects)
            self.sparse_retriever.add_documents(doc_objects)
            
            if save_path:
                dense_save_path = self._get_variant_index_path(save_path, "dense")
                sparse_save_path = self._get_variant_index_path(save_path, "sparse")
                self.dense_retriever.save(dense_save_path)
                self.sparse_retriever.save(sparse_save_path)
        else:
            self.retriever.add_documents(doc_objects)
            
            if save_path:
                self.retriever.save(save_path)
        
        logger.info(f"检索器索引已更新")

    def build_index(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """
        构建检索器索引

        Args:
            documents: 文档列表
            save_path: 保存路径
        """
        self.update_retriever_index(documents, save_path)