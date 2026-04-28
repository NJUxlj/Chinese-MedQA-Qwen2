# src/rag/rag_pipeline.py

import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from typing import Dict, List, Optional, Any
from knowledge_base.retrieval.milvus_retriever import MilvusRetriever
from knowledge_base.retrieval.bm25_retriever import BM25Retriever
from providers.embedding_provider import EmbeddingProvider
from providers.reranker_provider import RerankerProvider
from rag.query_processor import QueryProcessor
from rag.context_builder import ContextBuilder
from rag.response_generator import ResponseGenerator
from providers import LLMProvider
from utils.logger import setup_logger

from config.settings import settings

logger = setup_logger(__name__)

class RAGPipeline:
    """
    RAG流水线，整合查询处理、文档检索、上下文构建和响应生成
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config=None
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
        if config is None:
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                'retriever_type': settings.retriever.default_type if hasattr(settings.retriever, 'default_type') else 'hybrid',
                'embedding_model_name': settings.embedding.model_name if hasattr(settings.embedding, 'model_name') else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'embedding_model_path': settings.embedding.model_path if hasattr(settings.embedding, 'model_path') else None,
                'embedding_base_url': settings.embedding.base_url if hasattr(settings.embedding, 'base_url') else None,
                'embedding_api_key': settings.embedding.api_key if hasattr(settings.embedding, 'api_key') else None,
                'embedding_dimension': settings.embedding.dimension if hasattr(settings.embedding, 'dimension') else 768,
                'embedding_device': settings.embedding.device if hasattr(settings.embedding, 'device') else None,
                'index_path': None,
                'top_k': 3,
                'model_provider': settings.llm.model_provider if hasattr(settings.llm, 'model_provider') else 'openai',
                'model_name': settings.llm.model_name if hasattr(settings.llm, 'model_name') else 'Qwen3-235B',
                'model_path': settings.llm.model_path if hasattr(settings.llm, 'model_path') else None,
                'model_base_url': settings.llm.base_url if hasattr(settings.llm, 'base_url') else None,
                'model_api_key': settings.llm.api_key if hasattr(settings.llm, 'api_key') else None,
                'model_max_tokens': settings.llm.max_tokens if hasattr(settings.llm, 'max_tokens') else 2048,
                'model_temperature': settings.llm.temperature if hasattr(settings.llm, 'temperature') else 0.7,
                'cache_dir': None,
                'reranker_model_provider': 'transformers',
                'reranker_model_path': settings.reranker.model_path if hasattr(settings.reranker, 'model_path') else None,
                'reranker_model_name': settings.reranker.model_name if hasattr(settings.reranker, 'model_name') else None,
                'reranker_base_url': settings.reranker.base_url if hasattr(settings.reranker, 'base_url') else None,
                'reranker_api_key': settings.reranker.api_key if hasattr(settings.reranker, 'api_key') else None,
                'use_reranker': settings.reranker.use_reranker if hasattr(settings.reranker, 'use_reranker') else False,
                'hybrid_weight': settings.retriever.default_weight if hasattr(settings.retriever, 'default_weight') else 0.7,
                'chunk_size': 500,
                'chunk_overlap': 50,
                'response_template': None,
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
            })
        self.config = config


        self.retriever_type: str = self.config.retriever_type

        # 获取嵌入模型相关配置
        self.embedding_model_name: str = self.config.embedding_model_name
        self.embedding_model_path: Optional[str] = self.config.embedding_model_path if hasattr(self.config, 'embedding_model_path') else None
        self.embedding_base_url: Optional[str] = self.config.embedding_base_url if hasattr(self.config, 'embedding_base_url') else None
        self.embedding_api_key: Optional[str] = self.config.embedding_api_key if hasattr(self.config, 'embedding_api_key') else None
        self.embedding_dimension: int = self.config.embedding_dimension if hasattr(self.config, 'embedding_dimension') else 768
        self.embedding_device: Optional[str] = self.config.embedding_device if hasattr(self.config, 'embedding_device') else None

        self.index_path: Optional[str] = self.config.index_path if hasattr(self.config, 'index_path') else None
        self.top_k: int = self.config.top_k if hasattr(self.config, 'top_k') else 3

        # 获取 LLMProvider 相关配置
        self.model_provider: str = self.config.model_provider if hasattr(self.config, 'model_provider') else 'openai'
        self.model_name: str = self.config.model_name if hasattr(self.config, 'model_name') else 'Qwen3-235B'
        self.model_path: Optional[str] = self.config.model_path if hasattr(self.config, 'model_path') else None
        self.model_base_url: Optional[str] = self.config.model_base_url if hasattr(self.config, 'model_base_url') else None
        self.model_api_key: Optional[str] = self.config.model_api_key if hasattr(self.config, 'model_api_key') else None
        self.model_max_tokens: int = self.config.model_max_tokens if hasattr(self.config, 'model_max_tokens') else 2048
        self.model_temperature: float = self.config.model_temperature if hasattr(self.config, 'model_temperature') else 0.7


        self.cache_dir: Optional[str] = self.config.cache_dir if hasattr(self.config, 'cache_dir') else None

        # 获取重排序模型相关配置
        self.reranker_model_provider: str = self.config.reranker_model_provider if hasattr(self.config, 'reranker_model_provider') else 'transformers'
        self.reranker_model_path: Optional[str] = self.config.reranker_model_path if hasattr(self.config, 'reranker_model_path') else None
        self.reranker_model_name: Optional[str] = self.config.reranker_model_name if hasattr(self.config, 'reranker_model_name') else None
        self.reranker_base_url: Optional[str] = self.config.reranker_base_url if hasattr(self.config, 'reranker_base_url') else None
        self.reranker_api_key: Optional[str] = self.config.reranker_api_key if hasattr(self.config, 'reranker_api_key') else None
        self.use_reranker: bool = self.config.use_reranker if hasattr(self.config, 'use_reranker') else False

        self.hybrid_weight: float = self.config.hybrid_weight if hasattr(self.config, 'hybrid_weight') else 0.7
        self.chunk_size: int = self.config.chunk_size if hasattr(self.config, 'chunk_size') else 500
        self.chunk_overlap: int = self.config.chunk_overlap if hasattr(self.config, 'chunk_overlap') else 50
        self.response_template: Optional[str] = self.config.response_template if hasattr(self.config, 'response_template') else None
        self.max_new_tokens: int = self.config.max_new_tokens if hasattr(self.config, 'max_new_tokens') else 512
        self.temperature: float = self.config.temperature if hasattr(self.config, 'temperature') else 0.7
        self.top_p: float = self.config.top_p if hasattr(self.config, 'top_p') else 0.9

        self.collection_names: List[str] = self.config.collection_names if hasattr(self.config, 'collection_names') else ['medical_qa']

        # 初始化查询处理器
        self.query_processor = QueryProcessor()

        self.llm_provider = llm_provider

        # 初始化嵌入管理器 (对于某些检索器需要)
        if self.retriever_type in ["knn", "similarity", "l2", "hybrid"]:
            self.embedding_provider = EmbeddingProvider(
                model_name=self.embedding_model_name,
                model_path=self.embedding_model_path,
                base_url=self.embedding_base_url,
                api_key=self.embedding_api_key,
                device=self.embedding_device,
                cache_dir=self.cache_dir
            )
        else:
            self.embedding_provider = None

        # 根据类型初始化检索器（统一使用 MilvusRetriever，无需 faiss）
        if self.retriever_type in ["knn", "similarity", "l2"]:
            self.retriever = MilvusRetriever(
                embedding_provider=self.embedding_provider,
                collection_names=self.collection_names
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever()
        elif self.retriever_type == "hybrid":
            self.dense_retriever = MilvusRetriever(
                embedding_provider=self.embedding_provider,
                collection_names=self.collection_names
            )
            self.sparse_retriever = BM25Retriever()
            self.retriever = None
        else:
            raise ValueError(f"不支持的检索器类型: {self.retriever_type}")



        self.reranker_provider = RerankerProvider(
            model_provider=self.reranker_model_provider,
            model_path=self.reranker_model_path if self.use_reranker else None,
            model_name=self.reranker_model_name if self.use_reranker else None,
            base_url=self.reranker_base_url if self.use_reranker else None,
            api_key=self.reranker_api_key if self.use_reranker else None
        ) if self.use_reranker else None
        
        # 初始化上下文构建器
        self.context_builder = ContextBuilder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            reranker_provider=self.reranker_provider,
            max_context_length = 5000
        )
        

        self.response_generator = ResponseGenerator(
            llm_provider=self.llm_provider,
            max_tokens=self.model_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            template=self.response_template
        )

    
    
    

    
    def _query_hybrid(self, query_text: str, top_k: Optional[int] = None, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        使用混合检索方式进行查询

        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            filter: 过滤条件

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

        # 应用过滤器
        if filter:
            sorted_docs = self._apply_filter(sorted_docs, filter)

        return sorted_docs[:k]



    
    def query(self, query_text: str, top_k: Optional[int] = None, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关文档

        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            filter: 过滤条件，格式为 {"field": "value"}

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
            return self._query_hybrid(processed_query, k, filter=filter)

        results = self.retriever.search(processed_query, k)

        docs = []
        for doc, score in results:
            docs.append({
                "id": doc.metadata.get("doc_id", "unknown"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })

        # 应用过滤器
        if filter:
            docs = self._apply_filter(docs, filter)

        return docs

    def _apply_filter(self, docs: List[Dict[str, Any]], filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对文档列表应用过滤条件"""
        filtered = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            match = True
            for key, value in filter.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    
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
        
        # 构建上下文
        context = self.context_builder.build_context(
            query=query_text,
            documents=retrieved_docs,
        )
        
        return context
    
    def build_rag_prompt(
        self, 
        query_text: str, 
        top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        构建RAG提示
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            包含查询、上下文和格式化提示的字典
        """
        documents = self.query(query_text, top_k)

        if not documents:
            logger.warning("没有检索到相关文档")
            context = ""
        else:
            context = self.context_builder.build_context(
                query=query_text,
                documents=documents,
            )

        prompt = {
            "query": query_text,
            "context": context,
            "documents": documents,
            "formatted_prompt": self.context_builder.format_prompt(query_text, context)
        }

        return prompt
    
    def generate_response(
        self, 
        query_text: str, 
        top_k: Optional[int] = None) -> Dict[str, Any]:

        """
        生成RAG增强的响应
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            包含响应和元数据的字典
        """
        if not self.llm_provider or not self.response_generator:
            raise ValueError("未设置模型，无法生成响应")
        
        # 构建RAG提示
        rag_prompt = self.build_rag_prompt(query_text, top_k)
        
        # 生成响应
        response = self.response_generator.generate(rag_prompt)
        
        return response


        
    
