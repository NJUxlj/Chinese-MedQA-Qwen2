from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import torch
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from config.reranker_config import RerankerConfig
from langchain_core.documents import Document
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class RerankerService:
    """ reranker 服务
    
    使用 reranker 模型，根据对 query 的相似度， 对已有的 document 列表进行重排序

        - 用于从本地加载 Qwen3-reranker 这样的模型进行重排序
    """
    def __init__(self, config: Optional[RerankerConfig] = None):
        """初始化 reranker 服务"""
        self.config = config or RerankerConfig()
        self.model_provider = self.config.model_provider
        self.reranker_model = None
        self._load_reranker_model()


    def _load_reranker_model(self) -> None:
        """加载 reranker 模型"""
        model_path = self.config.model_path
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading reranker model from: {model_path}")
        
        if self.model_provider == "sentence_transformers":
            self.reranker_model = CrossEncoder(
                model_name=model_path,
                device=self.config.device,
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded reranker model: {self.config.model_name}")
        else:
            raise ValueError(f"Invalid model provider: {self.model_provider}. Currently only 'sentence_transformers' is supported.")


    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """ rerank 文档
        
        Args:
            query: 查询字符串
            documents: 待重排序的文档列表
            top_k: 返回的 top_k 个文档
            
        Returns:
            重排序后的文档列表
        """
        if not query:
            logger.warning("Query is empty, returning original documents")
            return documents[:top_k]
            
        if not documents:
            logger.warning("Documents list is empty")
            return []
        
        if self.reranker_model is None:
            raise ValueError("Reranker model not loaded")
        
        try:
            logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
            
            query_doc_pairs = [(query, doc.page_content) for doc in documents]
            
            scores = self.reranker_model.predict(
                query_doc_pairs,
                batch_size=self.config.batch_size
            )
            
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            if self.config.normalize_scores:
                min_score = scores.min()
                max_score = scores.max()
                if max_score - min_score > 0:
                    scores = (scores - min_score) / (max_score - min_score)
                else:
                    scores = scores - min_score
            
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_documents = [doc for doc, _ in doc_score_pairs[:top_k]]
            
            logger.info(f"Reranking completed. Top {top_k} documents selected.")
            
            self.print_reranked_documents_and_scores(doc_score_pairs[:top_k])
            
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            raise


    def print_reranked_documents_and_scores(self, doc_score_pairs: List[tuple]):
        """打印重排序后的文档和分数
        
        Args:
            doc_score_pairs: 文档和分数的元组列表
        """
        if not doc_score_pairs:
            logger.info("No documents to display")
            return
        
        logger.info("=" * 80)
        logger.info("Reranked Documents with Scores:")
        logger.info("=" * 80)
        
        for i, (doc, score) in enumerate(doc_score_pairs, 1):
            doc_preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"[{i}] Score: {score:.4f} | Preview: {doc_preview}...")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = RerankerConfig()
    reranker_service = RerankerService(config)
    
    
    test_query = "什么是高血压？"
    test_documents = [
        Document(page_content="高血压是一种常见的慢性疾病，特征是动脉血压持续升高。"),
        Document(page_content="糖尿病是一种代谢性疾病，影响人体对血糖的调节能力。"),
        Document(page_content="高血压患者应该注意饮食，减少盐分摄入。"),
        Document(page_content="心脏病是指心脏功能或结构的异常，可能导致心力衰竭。"),
        Document(page_content="适当的运动有助于控制血压水平。"),
    ]
    
    reranked_docs = reranker_service.rerank(test_query, test_documents, top_k=3)
    print(f"\nFinal reranked documents: {len(reranked_docs)}")
