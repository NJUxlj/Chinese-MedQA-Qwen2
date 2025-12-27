import os, sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Optional, Callable
from config.base_config import BaseConfig
from dotenv import load_dotenv

load_dotenv()


class RetrieverConfig(BaseConfig):
    """ retriever 配置
    
    """
    name: str = Field(default="bm25_retriever", description="Name of the retriever")  
    score_threshold: float = Field(default=0.1, description="Minimum BM25 score threshold")  
    use_jieba: bool = Field(default=True, description="Whether to use jieba for Chinese tokenization")  
    tokenizer: Optional[Callable[[str], List[str]]] = Field(default=None, description="Custom tokenizer function")  


class BM25RetrieverConfig(RetrieverConfig):
    """ BM25 retriever 配置
    
    """



class KNNRetrieverConfig(RetrieverConfig):
    """ KNN retriever 配置
    
    """
    name: str = Field(default="knn_retriever", description="Name of the retriever")  
    index_type: str = Field(default="Flat", description="FAISS index type: 'Flat', 'IVF', or 'HNSW'")  
    n_list: int = Field(default=100, description="Number of IVF clusters (for IVF index)")  
    m: int = Field(default=16, description="Number of connections per element (for HNSW index)")  


class L2RetrieverConfig(RetrieverConfig):
    """ L2 retriever 配置
    
    """
    distance_threshold: float = Field(default=0.5, description="Maximum L2 distance threshold (lower is better)")



class PubmedRetrieverConfig(RetrieverConfig):
    """ Pubmed retriever 配置
    
    """



class SimilarityRetrieverConfig(RetrieverConfig):
    """ Similarity retriever 配置
    
    """
