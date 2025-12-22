import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()


class RAGConfig(BaseModel):
    """ RAG 配置"""
    retriever_type: str = Field(default="knn", description="检索类型：knn, similarity, bm25, l2")
    embedding_model_name: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", description="嵌入模型名称")
    index_path: str = Field(default="data/indices/faiss_index.bin", description="索引路径")
    top_k: int = Field(default=5, description="默认检索数量")