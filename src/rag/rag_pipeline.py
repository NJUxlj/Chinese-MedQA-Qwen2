import os
from typing import Dict, List, Optional, Union, Any, Tuple
from knowledge_base.retrieval.knn_retriever import KNNRetriever
from knowledge_base.retrieval.similarity_retriever import SimilarityRetriever
from knowledge_base.retrieval.bm25_retriever import BM25Retriever
from knowledge_base.retrieval.l2_retriever import L2Retriever
from knowledge_base.embedding_manager import EmbeddingManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGPipeline:
    """RAG流水线，整合查询处理、文档检索和响应生成"""
    
    def __init__(
        self,
        retriever_type: str = "knn",  # "knn", "similarity", "bm25", "l2"
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dimension: int = 384,
        index_path: Optional[str] = None,
        top_k: int = 5,
        cache_dir: Optional[str] = None,
    ):
        self.retriever_type = retriever_type
        self.top_k = top_k
        
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(
            embedding_model_name_or_path=embedding_model_name,
            embedding_dimension=embedding_dimension,
            cache_dir=cache_dir
        )
        
        # 根据类型初始化检索器
        if retriever_type == "knn":
            self.retriever = KNNRetriever(
                embedding_manager=self.embedding_manager,
                index_path=index_path
            )
        elif retriever_type == "similarity":
            self.retriever = SimilarityRetriever(
                embedding_manager=self.embedding_manager,
                index_path=index_path
            )
        elif retriever_type == "bm25":
            self.retriever = BM25Retriever(
                index_path=index_path
            )
        elif retriever_type == "l2":
            self.retriever = L2Retriever(
                embedding_manager=self.embedding_manager,
                index_path=index_path
            )
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """根据查询文本检索相关文档"""
        k = top_k or self.top_k
        return self.retriever.retrieve(query_text, k)
    
    def build_context(self, query_text: str, top_k: Optional[int] = None) -> str:
        """构建增强上下文"""
        # 检索相关文档
        retrieved_docs = self.query(query_text, top_k)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            return ""
        
        # 构建上下文
        context = "以下是相关的医疗信息：\n\n"
        
        for i, doc in enumerate(retrieved_docs):
            context += f"[文档 {i+1}]: {doc['text']}\n\n"
        
        return context.strip()
    
    def build_rag_prompt(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """构建RAG提示"""
        context = self.build_context(query_text, top_k)
        
        prompt = {
            "query": query_text,
            "context": context,
            "formatted_prompt": f"请根据以下信息回答问题。如果无法从提供的信息中找到答案，请基于可靠的医学知识回答，并注明这是基于一般医学知识的回答。\n\n相关信息：\n{context}\n\n问题：{query_text}\n\n回答："
        }
        
        return prompt


# rag/response_generator.py

from typing import Dict, List, Optional, Union, Any, Tuple
from models.base_model import BaseModel
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ResponseGenerator:
    """响应生成器，负责生成RAG增强的最终回答"""
    
    def __init__(
        self,
        model: BaseModel,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """生成增强响应"""
        formatted_prompt = prompt.get("formatted_prompt", "")
        query = prompt.get("query", "")
        context = prompt.get("context", "")
        
        if not formatted_prompt:
            logger.error("Empty prompt passed to generator")
            return {"response": "抱歉，我无法处理空的提示。", "error": "Empty prompt"}
        
        try:
            # 使用模型生成回答
            response = self.model.generate(
                formatted_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            result = {
                "query": query,
                "context": context,
                "prompt": formatted_prompt,
                "response": response,
                "source_documents": prompt.get("source_documents", [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generating response: {e}")
            return {"response": f"生成回答时出错: {str(e)}", "error": str(e)}

