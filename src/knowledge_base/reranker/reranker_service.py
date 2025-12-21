from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import spacy
from typing import List, Dict, Any
from config.reranker_config import RerankerConfig



class RerankerService:
    """ reranker 服务
    
    使用 reranker 模型，根据对 query 的相似度， 对已有的 document 列表进行重排序
    """
    def __init__(self, config: RerankerConfig):
        """初始化 reranker 服务"""
        self.config = config
        self.model_provider = config.model_provider
        self.reranker_model = None
        self._load_reranker_model()


    

    def _load_reranker_model(self) -> None:
        """加载 reranker 模型"""
        if model_provider == "transformers":
            pass
        else:
            raise ValueError(f"Invalid model provider: {model_provider}")



    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """ rerank 文档"""
        pass





if __name__ == "__main__":
    config = RerankerConfig()
    reranker_service = RerankerService(config)