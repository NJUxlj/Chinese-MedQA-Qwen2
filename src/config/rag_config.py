import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseGenerativeModel

from dotenv import load_dotenv

load_dotenv()


class RAGConfig(BaseModel):
    """ RAG 配置"""
    retriever_type: str = Field(default="knn", description="检索类型：knn, similarity, bm25, l2")
    embedding_model_name: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", description="嵌入模型名称")
    index_path: str = Field(default="data/indices/faiss_index.bin", description="索引路径")
    top_k: int = Field(default=5, description="默认检索数量")

    model: Optional[BaseGenerativeModel] = Field(default=None, description="生成模型")
    cache_dir: Optional[str] = Field(default=None, description="缓存目录")
    reranker_model_name: Optional[str] = Field(default=None, description="重排序模型名称")
    use_reranker: bool = Field(default=False, description="是否使用重排序模型")
    hybrid_weight: float = Field(default=0.5, description="混合检索时的权重(dense:sparse)")
    chunk_size: int = Field(default=500, description="上下文构建时的块大小")
    chunk_overlap: int = Field(default=100, description="上下文块重叠大小")
    response_template: Optional[str] = Field(default=None, description="响应模板")
    max_new_tokens: int = Field(default=1024, description="生成的最大token数量")
    temperature: float = Field(default=0.7, description="生成温度")
    top_p: float = Field(default=0.9, description="生成top_p值")
