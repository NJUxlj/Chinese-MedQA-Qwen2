"""
Milvus Retriever - 基于 Milvus 向量数据库的检索器
替换 FAISS 的 KNN/Similarity/L2 检索器，统一使用 Milvus 作为向量存储后端
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document

from knowledge_base.retrieval.base_retriever import BaseRetriever
from providers.embedding_provider import EmbeddingProvider
from config.settings import settings
from knowledge_base.milvus.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class MilvusRetriever(BaseRetriever):
    """
    基于 Milvus 的检索器，支持从多个 Collection 进行向量相似度搜索。

    替换原来的 KNNRetriever / SimilarityRetriever / L2Retriever，
    统一使用 Milvus 作为向量数据库后端，无需 faiss。
    检索时同时查询多个 collection（web_crawled_texts, text_books, guidelines），
    并按分数合并结果。
    """

    def __init__(
        self,
        config=None,
        embedding_provider: EmbeddingProvider = None,
        collection_names: List[str] = None,
    ):
        """
        初始化 Milvus 检索器。

        Args:
            config: 检索器配置
            embedding_provider: 嵌入提供者（用于生成查询向量）
            collection_names: Milvus Collection 名称列表，默认使用 config 中的 3 个
        """
        super().__init__(config=config)

        self.embedding_provider = embedding_provider or EmbeddingProvider()

        # 默认使用 config 中配置的 3 个 collection
        self.collection_names = collection_names or (
            list(settings.milvus.collection_names)
            if hasattr(settings.milvus, 'collection_names') and settings.milvus.collection_names
            else ["web_crawled_texts", "text_books", "guidelines"]
        )

        # 初始化 Milvus 客户端
        self.milvus_client = MilvusClient(
            milvus_config=settings.milvus,
            embedding_config=settings.embedding,
        )
        self._connected = False

    def connect(self) -> bool:
        """连接到 Milvus 服务器"""
        if not self._connected:
            self._connected = self.milvus_client.connect()
        return self._connected

    def add_documents(self, documents: List[Document], collection_name: str = None) -> None:
        """
        添加文档到指定 Milvus Collection。

        Args:
            documents: 文档列表
            collection_name: Collection 名称，默认使用第一个
        """
        if not documents:
            return

        target = collection_name or (self.collection_names[0] if self.collection_names else "medical_kb")

        if not self.connect():
            logger.error("无法连接到 Milvus，添加文档失败")
            return

        ids = [doc.metadata.get("doc_id", str(i)) for i, doc in enumerate(documents)]
        self.milvus_client.add_documents(
            collection_name=target,
            documents=documents,
            ids=ids,
        )
        logger.info(f"已将 {len(documents)} 个文档添加到 Milvus Collection: {target}")

    def delete_documents(self, document_ids: List[str], collection_name: str = None) -> None:
        """
        从指定 Milvus Collection 删除文档。

        Args:
            document_ids: 要删除的文档 ID 列表
            collection_name: Collection 名称
        """
        target = collection_name or (self.collection_names[0] if self.collection_names else "medical_kb")

        if not self.connect():
            logger.error("无法连接到 Milvus，删除文档失败")
            return

        self.milvus_client.delete_documents(
            collection_name=target,
            ids=document_ids,
        )
        logger.info(f"已从 Milvus 删除 {len(document_ids)} 个文档")

    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        执行相似度搜索（同时查询所有 collection 并合并结果）。

        Args:
            query: 查询文本
            top_k: 返回结果总数量
            **kwargs: 其他参数

        Returns:
            List[Tuple[Document, float]]: (文档, 分数) 列表，按分数降序
        """
        if not self.connect():
            logger.error("无法连接到 Milvus，搜索失败")
            return []

        all_results: List[Tuple[Document, float]] = []

        for collection in self.collection_names:
            try:
                results = self.milvus_client.similarity_search_with_score(
                    collection_name=collection,
                    query=query,
                    k=top_k * 2,  # 每个 collection 多取一些，留出合并空间
                )
                if results:
                    # 为结果标注来源 collection
                    for doc, score in results:
                        doc.metadata["collection"] = collection
                    all_results.extend(results)
            except Exception as e:
                logger.warning(f"从 collection {collection} 搜索失败: {e}")
                continue

        # 按分数降序排序，保留 top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        final_results = all_results[:top_k]

        logger.info(f"Milvus 多-collection 搜索完成，共返回 {len(final_results)} 个结果")
        return final_results

    def save(self, directory: str) -> None:
        """保存（Milvus 数据持久化在服务器端）"""
        logger.info(f"Milvus 数据存储在服务器端，无需本地保存")

    def load(self, directory: str) -> None:
        """加载（Milvus 连接由构造时建立）"""
        logger.info(f"Milvus 连接由构造时建立")
