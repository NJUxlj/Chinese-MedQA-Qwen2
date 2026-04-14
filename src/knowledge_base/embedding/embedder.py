from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings

class TextEmbedder:
    """文本嵌入器"""

    def __init__(self, config=None):
        """初始化嵌入器"""
        self.config = config if config is not None else settings.embedding
        self.embedder = HuggingFaceEmbeddings(model_name=self.config.model_name)



    def embed_text(self, text: str) -> List[float]:
        """将文本转换为嵌入向量"""
        return self.embedder.embed_query(text)


    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """将文档列表转换为嵌入向量列表"""
        return self.embedder.embed_documents(documents)

    def embed_large_documents(self, documents: List[str], batch_size: int = 100) -> List[List[float]]:
        """将大型文档列表转换为嵌入向量列表，支持批量处理"""
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings.extend(self.embed_documents(batch))
        return embeddings