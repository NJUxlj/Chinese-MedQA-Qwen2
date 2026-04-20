"""
嵌入服务
负责文本嵌入和向量操作
"""

from typing import Dict, Any, List, Optional, Union
import sys
import threading
import logging
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings
from knowledge_base.embedding.embedding_manager import EmbeddingManager
from utils.logger import setup_logger


class EmbeddingService:
    """嵌入服务类"""

    def __init__(self):
        """初始化嵌入服务"""
        self.embedding_managers: Dict[str, EmbeddingManager] = {}
        self._managers_lock = threading.RLock()
        self.logger = setup_logger(self.__class__.__name__)

        cfg = settings.embedding_service
        self.default_model_name = str(cfg.default_model_name)
        self.default_dimension = int(cfg.default_dimension)
        self.cache_dir = str(cfg.cache_dir)
    
    def load_default_model(self) -> EmbeddingManager:
        """
        加载默认嵌入模型
        
        Returns:
            嵌入管理器实例
        """
        return self.get_embedding_manager(self.default_model_name, self.default_dimension)
    
    def get_embedding_manager(self, model_name: str, 
                              dimension: Optional[int] = None) -> EmbeddingManager:
        """
        获取嵌入管理器
        
        Args:
            model_name: 模型名称
            dimension: 嵌入维度
            
        Returns:
            嵌入管理器实例
        """
        with self._managers_lock:
            # 检查是否已加载该模型
            if model_name in self.embedding_managers:
                return self.embedding_managers[model_name]
            
            self.logger.info(f"加载嵌入模型: {model_name}")
            
            try:
                # 如果未指定维度，使用默认维度
                if dimension is None:
                    dimension = self.default_dimension
                
                # 创建嵌入管理器
                manager = EmbeddingManager(
                    embedding_model_name_or_path=model_name,
                    embedding_dimension=dimension,
                    cache_dir=self.cache_dir
                )
                
                # 保存管理器实例
                self.embedding_managers[model_name] = manager
                
                return manager
                
            except Exception as e:
                self.logger.error(f"加载嵌入模型 {model_name} 失败: {e}")
                raise
    
    def get_default_embedding_manager(self) -> EmbeddingManager:
        """
        获取默认嵌入管理器
        
        Returns:
            默认嵌入管理器实例
        """
        return self.get_embedding_manager(self.default_model_name, self.default_dimension)
    
    def get_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        获取文本嵌入
        
        Args:
            text: 输入文本
            model_name: 模型名称
            
        Returns:
            嵌入向量
        """
        # 如果未指定模型，使用默认模型
        if model_name is None:
            model_name = self.default_model_name
        
        # 获取嵌入管理器
        manager = self.get_embedding_manager(model_name)
        
        # 获取嵌入
        embedding = manager.get_embedding(text)
        
        return embedding.tolist()
    
    def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        获取多个文本的嵌入
        
        Args:
            texts: 输入文本列表
            model_name: 模型名称
            
        Returns:
            嵌入向量列表
        """
        # 如果未指定模型，使用默认模型
        if model_name is None:
            model_name = self.default_model_name
        
        # 获取嵌入管理器
        manager = self.get_embedding_manager(model_name)
        
        # 获取嵌入
        embeddings = manager.get_embeddings(texts)
        
        return [emb.tolist() for emb in embeddings]
    
    def calculate_similarity(self, text1: str, text2: str, 
                           model_name: Optional[str] = None) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            model_name: 模型名称
            
        Returns:
            相似度分数(0-1)
        """
        # 如果未指定模型，使用默认模型
        if model_name is None:
            model_name = self.default_model_name
        
        # 获取嵌入管理器
        manager = self.get_embedding_manager(model_name)
        
        # 获取嵌入
        embedding1 = manager.get_embedding(text1)
        embedding2 = manager.get_embedding(text2)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的嵌入模型列表
        
        Returns:
            模型名称列表
        """
        with self._managers_lock:
            return list(self.embedding_managers.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载嵌入模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功卸载
        """
        with self._managers_lock:
            if model_name not in self.embedding_managers:
                self.logger.warning(f"嵌入模型 {model_name} 未加载，无法卸载")
                return False
            
            self.logger.info(f"卸载嵌入模型: {model_name}")
            
            try:
                # 从字典中移除
                del self.embedding_managers[model_name]
                return True
            
            except Exception as e:
                self.logger.error(f"卸载嵌入模型 {model_name} 失败: {e}")
                return False

# 单例模式
_embedding_service = None
_lock = threading.Lock()

def get_embedding_service() -> EmbeddingService:
    """
    获取嵌入服务单例
    
    Returns:
        嵌入服务实例
    """
    global _embedding_service
    
    if _embedding_service is None:
        with _lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService()
    
    return _embedding_service