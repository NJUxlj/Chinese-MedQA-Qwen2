"""
提供 Milvus 向量数据库的基础操作功能，包括：
- 数据库和 Collection 管理
- 文档增删改查
- 相似度搜索和混合搜索
- 索引管理

依赖安装：
pip install langchain-milvus pymilvus langchain-openai
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple
from uuid import uuid4

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from pymilvus.exceptions import MilvusException
from pymilvus import utility

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger
from providers.embedding_provider import EmbeddingProvider
from config.settings import settings

try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


class MilvusClient:
    """
    基于 LangChain 的 Milvus 客户端类

    提供 Milvus 向量数据库的基础操作功能，包括数据库管理、
    Collection 管理、文档操作和搜索功能。
    """

    def __init__(
        self,
        milvus_config=None,
        embedding_config=None
    ):
        """
        初始化 Milvus 客户端

        Args:
            uri: Milvus 服务器 URI
            token: Milvus 认证令牌
            db_name: 数据库名称
            embedding_model: 嵌入模型名称
            embedding_api_key: 嵌入模型 API 密钥
            index_type: 索引类型 (FLAT, IVF, HNSW 等)
            metric_type: 距离度量类型 (L2, IP, COSINE)
            consistency_level: 一致性级别
        """
        self.milvus_config = milvus_config if milvus_config is not None else settings.milvus
        self.embedding_config = embedding_config if embedding_config is not None else settings.embedding

        self.uri = self.milvus_config.uri
        self.token = self.milvus_config.token
        self.db_name = self.milvus_config.db_name
        self.index_type = self.milvus_config.index_type
        self.metric_type = self.milvus_config.metric_type
        self.consistency_level = self.milvus_config.consistency_level
        
        # 初始化嵌入函数（优先使用本地 model_path，避免网络请求）
        _model_path = getattr(self.embedding_config, "model_path", None)
        _model_name = str(self.embedding_config.model_name)
        self.embedder = EmbeddingProvider(
            mode="local",
            model_name=_model_name,
            model_path=str(_model_path) if _model_path and str(_model_path) not in ("None", "null", "") else None,
            device="cuda" if _CUDA_AVAILABLE else "cpu",
        )

        # 初始化 Milvus 向量存储
        self.vector_store = None
        
        # 连接状态
        self._connected = False

        self.logger = setup_logger(self.__class__.__name__, level="INFO")
    
    
    def connect(self) -> bool:
        """
        连接到 Milvus 服务器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            connections.connect(
                uri=self.uri,
                token=self.token,
                db_name=self.db_name
            )
            self._connected = True
            self.logger.info(f"成功连接到 Milvus 服务器: {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"连接 Milvus 失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开与 Milvus 服务器的连接"""
        try:
            connections.disconnect("default")
            self._connected = False
            self.logger.info("已断开与 Milvus 服务器的连接")
        except Exception as e:
            self.logger.error(f"断开连接失败: {e}")
    
    def create_database(self, database_name: str) -> bool:
        """
        创建数据库
        
        Args:
            database_name: 数据库名称
            
        Returns:
            bool: 创建是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        try:
            utility.create_database(database_name)
            self.logger.info(f"成功创建数据库: {database_name}")
            return True
        except Exception as e:
            self.logger.error(f"创建数据库失败:  {e}")
            return False
    
    def list_databases(self) -> List[str]:
        """
        列出所有数据库
        
        Returns:
            List[str]: 数据库名称列表
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return []

        try:
            databases = utility.list_database()
            return [db for db in databases]
        except Exception as e:
            self.logger.error(f"列出数据库失败: {e}")
            return []
    
    def database_exists(self, database_name: str) -> bool:
        """
        检查数据库是否存在
        
        Args:
            database_name: 数据库名称
            
        Returns:
            bool: 数据库是否存在
        """
        databases = self.list_databases()
        return database_name in databases
    
    def create_collection(
        self,
        collection_name: str,
        vector_field_name: str = "vector",
        text_field_name: str = "text",
        metadata_field_name: str = "metadata",
        dimension: Optional[int] = None,
        drop_existing: bool = False
    ) -> bool:
        """
        创建 Collection
        
        Args:
            collection_name: Collection 名称
            vector_field_name: 向量字段名称
            text_field_name: 文本字段名称
            metadata_field_name: 元数据字段名称
            dimension: 向量维度，如果为 None 则自动检测
            drop_existing: 是否删除已存在的 Collection
            
        Returns:
            bool: 创建是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        try:
            # 如果指定了 drop_existing，先删除
            if drop_existing and self.collection_exists(collection_name):
                self.drop_collection(collection_name)
            
            # 获取向量维度
            if dimension is None:
                test_vector = self.embedder.embed_query("test")
                dimension = len(test_vector)
            
            # 创建字段定义
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=True,
                    max_length=100
                ),
                FieldSchema(
                    name=vector_field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension
                ),
                FieldSchema(
                    name=text_field_name,
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name=metadata_field_name,
                    dtype=DataType.JSON
                )
            ]
            
            # 创建 Collection Schema
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            
            # 创建 Collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                consistency_level=self.consistency_level
            )
            
            # 创建索引
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": 1024} if self.index_type.startswith("IVF") else {}
            }
            
            collection.create_index(vector_field_name, index_params)
            collection.load()
            
            self.logger.info(f"成功创建 Collection: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建 Collection 失败: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        列出所有 Collection
        
        Returns:
            List[str]: Collection 名称列表
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return []
        self.logger.info("正在列出所有 Collection...")
        try:
            collections = utility.list_collections()
            return list(collections)
        except Exception as e:
            self.logger.error(f"列出 Collection 失败: {e}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        检查 Collection 是否存在
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: Collection 是否存在
        """
        collections = self.list_collections()
        return collection_name in collections
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        删除 Collection
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: 删除是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        if not self.collection_exists(collection_name):
            self.logger.warning(f"Collection {collection_name} 不存在")
            return True
        
        try:
            collection = Collection(collection_name)
            collection.drop()
            self.logger.info(f"成功删除 Collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"删除 Collection 失败: {e}")
            return False
    
    def init_vector_store(self, collection_name: str) -> bool:
        """
        初始化向量存储
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.vector_store = Milvus(
                embedding_function=self.embedder,
                connection_args={
                    "uri": self.uri,
                    "token": self.token,
                    "db_name": self.db_name
                },
                collection_name=collection_name,
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type
                },
                consistency_level=self.consistency_level
            )
            self.logger.info(f"成功初始化向量存储: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"初始化向量存储失败: {e}")
            return False
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        向指定 Collection 添加文档
        
        Args:
            collection_name: Collection 名称
            documents: 文档列表
            ids: 文档 ID 列表，如果为 None 则自动生成
            
        Returns:
            List[str]: 添加的文档 ID 列表
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return []
        
        # 初始化向量存储
        if not self.init_vector_store(collection_name):
            return []
        
        try:
            # 生成 ID
            if ids is None:
                ids = [str(uuid4()) for _ in documents]
            
            # 添加文档
            self.vector_store.add_documents(documents=documents, ids=ids)
            self.logger.info(f"成功添加 {len(documents)} 个文档到 Collection: {collection_name}")
            return ids
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            return []
    
    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> bool:
        """
        删除文档
        
        Args:
            collection_name: Collection 名称
            ids: 要删除的文档 ID 列表
            filter_expr: 过滤表达式
            
        Returns:
            bool: 删除是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        try:
            # 初始化向量存储
            if not self.init_vector_store(collection_name):
                return False
            
            if ids:
                self.vector_store.delete(ids=ids)
            elif filter_expr:
                collection = Collection(collection_name)
                collection.delete(filter_expr)
            else:
                self.logger.error("必须指定要删除的文档 ID 或过滤条件")
                return False
            
            self.logger.info(f"成功删除文档从 Collection: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 4,
        filter_expr: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            collection_name: Collection 名称
            query: 查询文本
            k: 返回结果数量
            filter_expr: 过滤表达式
            score_threshold: 分数阈值
            
        Returns:
            List[Document]: 搜索结果文档列表
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return []
        
        try:
            # 初始化向量存储
            if not self.init_vector_store(collection_name):
                return []
            
            # 执行搜索
            search_kwargs = {"k": k}
            if filter_expr:
                search_kwargs["expr"] = filter_expr
            if score_threshold > 0:
                search_kwargs["score_threshold"] = score_threshold
            
            results = self.vector_store.similarity_search(query, **search_kwargs)
            self.logger.info(f"相似度搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"相似度搜索失败: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        collection_name: str,
        query: str,
        k: int = 4,
        filter_expr: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索（带分数）
        
        Args:
            collection_name: Collection 名称
            query: 查询文本
            k: 返回结果数量
            filter_expr: 过滤表达式
            score_threshold: 分数阈值
            
        Returns:
            List[Tuple[Document, float]]: 搜索结果和分数列表
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return []
        
        try:
            # 初始化向量存储
            if not self.init_vector_store(collection_name):
                return []
            
            # 执行搜索
            search_kwargs = {"k": k}
            if filter_expr:
                search_kwargs["expr"] = filter_expr
            if score_threshold > 0:
                search_kwargs["score_threshold"] = score_threshold
            
            results = self.vector_store.similarity_search_with_score(query, **search_kwargs)
            self.logger.info(f"相似度搜索（带分数）完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"相似度搜索（带分数）失败: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        获取 Collection 信息
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            Dict[str, Any]: Collection 信息
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return {}
        
        try:
            collection = Collection(collection_name)
            info = {
                "name": collection_name,
                "schema": collection.schema,
                "num_entities": collection.num_entities,
                "indexes": collection.indexes,
                "description": collection.description
            }
            return info
        except Exception as e:
            self.logger.error(f"获取 Collection 信息失败: {e}")
            return {}
    
    def flush_collection(self, collection_name: str) -> bool:
        """
        刷新 Collection（将内存中的数据写入磁盘）
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: 刷新是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        try:
            collection = Collection(collection_name)
            collection.flush()
            self.logger.info(f"成功刷新 Collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"刷新 Collection 失败: {e}")
            return False
    
    def compact_collection(self, collection_name: str) -> bool:
        """
        压缩 Collection（清理已删除的数据）
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: 压缩是否成功
        """
        if not self._connected:
            self.logger.error("请先连接到 Milvus 服务器")
            return False
        
        try:
            collection = Collection(collection_name)
            collection.compact()
            self.logger.info(f"成功压缩 Collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"压缩 Collection 失败: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


# 使用示例
if __name__ == "__main__":
    # 使用示例
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus",
        db_name="default"
    )
    
    with client:
        # 创建数据库
        client.create_database("my_database")
        
        # 创建 Collection
        client.create_collection("my_collection", drop_existing=True)
        
        # 添加文档
        documents = [
            Document(page_content="这是第一篇文档", metadata={"source": "doc1"}),
            Document(page_content="这是第二篇文档", metadata={"source": "doc2"}),
            Document(page_content="这是第三篇文档", metadata={"source": "doc3"})
        ]
        
        ids = client.add_documents("my_collection", documents)
        print(f"添加的文档 ID: {ids}")
        
        # 相似度搜索
        results = client.similarity_search("my_collection", "文档", k=2)
        for i, doc in enumerate(results):
            print(f"结果 {i+1}: {doc.page_content}")
        
        # 带分数的搜索
        results_with_score = client.similarity_search_with_score("my_collection", "文档", k=2)
        for i, (doc, score) in enumerate(results_with_score):
            print(f"结果 {i+1} (分数: {score:.4f}): {doc.page_content}")



