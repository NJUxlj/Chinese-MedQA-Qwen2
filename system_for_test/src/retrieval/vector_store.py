
import os
from typing import Dict, List, Optional, Any, Union
import hashlib

import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema.embeddings import Embeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document

class MedicalVectorStore:
    """医疗向量存储器，用于存储和检索文档向量表示"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化医疗向量存储器。
        
        Args:
            config: 向量存储配置字典
        """
        self.config = config
        self.provider = config.get('provider', 'chroma')
        self.embedding_model_name = config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
        self.collection_name = config.get('collection_name', 'medical_knowledge')
        self.persist_directory = config.get('persist_directory', './data/vector_db')
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # 创建存储目录
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化向量存储
        self._init_vector_store()
        
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            if self.provider.lower() == 'chroma':
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
            elif self.provider.lower() == 'faiss':
                # 对于FAISS，我们首先检查是否存在现有索引
                index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                if os.path.exists(index_path):
                    self.vector_store = FAISS.load_local(
                        self.persist_directory,
                        self.embedding_model,
                        self.collection_name
                    )
                else:
                    # 创建一个空的FAISS索引
                    self.vector_store = FAISS(
                        embedding_function=self.embedding_model,
                        index_name=self.collection_name
                    )
            else:
                raise ValueError(f"不支持的向量存储提供者: {self.provider}")
                
            print(f"成功初始化向量存储: {self.provider}")
            
        except Exception as e:
            print(f"初始化向量存储时出错: {e}")
            raise
            
    def add_documents(self, documents: List[Document]):
        """
        将文档添加到向量存储。
        
        Args:
            documents: 要添加的文档列表
        """
        try:
            # 为每个文档生成一个唯一ID
            ids = [self._generate_document_id(doc) for doc in documents]
            
            # 添加文档到向量存储
            self.vector_store.add_documents(documents=documents, ids=ids)
            
            # 对于Chroma，我们需要显式持久化
            if self.provider.lower() == 'chroma':
                self.vector_store.persist()
                
            # 对于FAISS，我们需要保存到磁盘
            elif self.provider.lower() == 'faiss':
                self.vector_store.save_local(self.persist_directory)
                
            print(f"成功添加 {len(documents)} 个文档到向量存储")
            
        except Exception as e:
            print(f"添加文档到向量存储时出错: {e}")
            raise
            
    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        使用语义搜索查询向量存储。
        
        Args:
            query: 搜索查询字符串
            k: 返回的结果数量
            filter: 元数据过滤条件（可选）
            
        Returns:
            相关文档列表，按相关性排序
        """
        try:
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            return docs
            
        except Exception as e:
            print(f"搜索向量存储时出错: {e}")
            return []
            
    def search_with_score(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        使用语义搜索查询向量存储，并返回相似度分数。
        
        Args:
            query: 搜索查询字符串
            k: 返回的结果数量
            filter: 元数据过滤条件（可选）
            
        Returns:
            (文档, 相似度分数)的元组列表，按相关性排序
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            return results
            
        except Exception as e:
            print(f"带分数搜索向量存储时出错: {e}")
            return []
            
    def clear(self):
        """清空向量存储（谨慎使用）"""
        try:
            if self.provider.lower() == 'chroma':
                self.vector_store.delete_collection()
                self._init_vector_store()
            elif self.provider.lower() == 'faiss':
                # 对于FAISS，我们需要重新创建索引
                index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                if os.path.exists(index_path):
                    os.remove(index_path)
                    
                # 重新初始化
                self._init_vector_store()
                
            print("成功清空向量存储")
            
        except Exception as e:
            print(f"清空向量存储时出错: {e}")
            raise
            
    def _generate_document_id(self, document: Document) -> str:
        """
        为文档生成唯一ID。
        
        Args:
            document: 文档对象
            
        Returns:
            文档的唯一ID字符串
        """
        # 组合文档内容和元数据来创建唯一ID
        content = document.page_content
        metadata_str = str(document.metadata)
        
        # 创建哈希
        doc_hash = hashlib.md5(f"{content}{metadata_str}".encode()).hexdigest()
        return doc_hash
