"""
模型服务
负责加载、管理和调用LLM模型
"""

from typing import Dict, Any, List, Optional, Union
import os
import threading
import logging
from contextlib import contextmanager

# 导入模型
from models.qwen_model import Qwen2Model
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        """初始化模型服务"""
        self.models: Dict[str, BaseModel] = {}
        self._models_lock = threading.RLock()  # 线程安全的读写锁
        
        # 默认模型配置
        self.default_model_name = os.environ.get("DEFAULT_MODEL", "qwen2-7b-instruct")
        self.default_model_path = os.environ.get("DEFAULT_MODEL_PATH", "Qwen/Qwen2-7B-Instruct")
    
    def load_model(self, model_name: str, model_path: str, 
                  model_type: str = "qwen2", **kwargs) -> BaseModel:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            model_type: 模型类型
            **kwargs: 其他参数
            
        Returns:
            加载的模型实例
        """
        with self._models_lock:
            # 检查模型是否已加载
            if model_name in self.models:
                logger.info(f"模型 {model_name} 已加载")
                return self.models[model_name]
            
            logger.info(f"开始加载模型: {model_name}, 路径: {model_path}")
            
            try:
                # 根据模型类型加载不同的模型
                if model_type.lower() == "qwen2":
                    model = Qwen2Model(
                        model_name=model_name,
                        model_path=model_path,
                        **kwargs
                    )
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                
                # 保存模型实例
                self.models[model_name] = model
                logger.info(f"模型 {model_name} 加载成功")
                return model
                
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {e}")
                raise
    
    def load_default_model(self) -> BaseModel:
        """
        加载默认模型
        
        Returns:
            默认模型实例
        """
        return self.load_model(
            model_name=self.default_model_name,
            model_path=self.default_model_path
        )
    
    def get_model(self, model_name: Optional[str] = None) -> BaseModel:
        """
        获取模型实例
        
        Args:
            model_name: 模型名称，如果为None则使用默认模型
            
        Returns:
            模型实例
        """
        with self._models_lock:
            if model_name is None:
                model_name = self.default_model_name
            
            # 检查模型是否已加载
            if model_name not in self.models:
                if model_name == self.default_model_name:
                    return self.load_default_model()
                else:
                    raise ValueError(f"模型 {model_name} 未加载")
            
            return self.models[model_name]
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功卸载
        """
        with self._models_lock:
            if model_name not in self.models:
                logger.warning(f"模型 {model_name} 未加载，无法卸载")
                return False
            
            logger.info(f"卸载模型: {model_name}")
            
            try:
                # 释放模型资源
                model = self.models[model_name]
                model.unload()
                
                # 从字典中移除
                del self.models[model_name]
                return True
            
            except Exception as e:
                logger.error(f"卸载模型 {model_name} 失败: {e}")
                return False
    
    def unload_all_models(self) -> None:
        """卸载所有模型"""
        with self._models_lock:
            model_names = list(self.models.keys())
            
            for model_name in model_names:
                self.unload_model(model_name)
    
    def get_loaded_models(self) -> List[str]:
        """
        获取已加载的模型列表
        
        Returns:
            模型名称列表
        """
        with self._models_lock:
            return list(self.models.keys())
    
    @contextmanager
    def get_model_context(self, model_name: Optional[str] = None):
        """
        获取模型上下文，用于临时使用模型
        
        Args:
            model_name: 模型名称
            
        Yields:
            模型实例
        """
        model = self.get_model(model_name)
        try:
            yield model
        finally:
            pass  # 可以添加清理逻辑

# 单例模式
_model_service = None
_lock = threading.Lock()

def get_model_service() -> ModelService:
    """
    获取模型服务单例
    
    Returns:
        模型服务实例
    """
    global _model_service
    
    if _model_service is None:
        with _lock:
            if _model_service is None:
                _model_service = ModelService()
    
    return _model_service