
# utils/logger.py

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Optional, Dict, Any

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# 默认日志配置
DEFAULT_LOG_CONFIG = {
    "level": "info",
    "format": DEFAULT_LOG_FORMAT,
    "log_dir": "logs",
    "max_bytes": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
    "console_output": True,
    "file_output": True
}


class LoggerManager:
    """日志管理器，处理日志记录器的创建和配置
    
    核心特性：
    - 单例模式，确保全局只有一个日志管理器实例
    - 支持控制台和文件双重输出
    - 自动按日期分割日志文件
    - 可配置日志级别、格式、文件大小等参数
    """
    
    _instance = None  # 单例实例
    _loggers: Dict[str, logging.Logger] = {}  # 已创建的日志记录器缓存
    _config: Dict[str, Any] = {}  # 全局日志配置
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现：确保全局只有一个LoggerManager实例"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化日志管理器
        
        Args:
            config: 自定义日志配置，覆盖默认配置
        """
        if self._initialized:
            return
        
        # 合并默认配置和自定义配置
        self._config = DEFAULT_LOG_CONFIG.copy()
        if config:
            self._config.update(config)
        
        # 确保日志目录存在
        if self._config["file_output"]:
            os.makedirs(self._config["log_dir"], exist_ok=True)
        
        self._initialized = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取或创建指定名称的日志记录器
        
        Args:
            name: 日志记录器名称，通常为模块名
        
        Returns:
            配置好的logging.Logger实例
        """
        # 检查是否已存在该名称的日志记录器
        if name in self._loggers:
            return self._loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        level = LOG_LEVELS.get(self._config["level"].lower(), logging.INFO)
        logger.setLevel(level)
        
        # 防止日志重复输出到父级记录器
        logger.propagate = False
        
        # 清除已有的处理器，避免重复配置
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建日志格式器
        log_format = self._config.get("format", DEFAULT_LOG_FORMAT)
        formatter = logging.Formatter(log_format)
        
        # 添加控制台输出处理器
        if self._config["console_output"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件输出处理器（按日期分割）
        if self._config["file_output"]:
            log_file = os.path.join(
                self._config["log_dir"], 
                f"{name}_{time.strftime('%Y%m%d')}.log"
            )
            
            # 使用RotatingFileHandler实现日志文件滚动
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self._config["max_bytes"],
                backupCount=self._config["backup_count"]
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 缓存日志记录器实例
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: str):
        """全局设置日志级别
        
        Args:
            level: 日志级别字符串，可选值：debug, info, warning, error, critical
        """
        if level.lower() not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}")
        
        self._config["level"] = level.lower()
        level_value = LOG_LEVELS[level.lower()]
        
        # 更新所有已创建的日志记录器级别
        for logger in self._loggers.values():
            logger.setLevel(level_value)
    
    def update_config(self, config: Dict[str, Any]):
        """更新全局日志配置
        
        Args:
            config: 新的配置参数
        """
        self._config.update(config)
        
        # 重置所有已创建的日志记录器，下次get_logger时会重新配置
        for name in list(self._loggers.keys()):
            self._loggers.pop(name)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    detailed: bool = False
) -> logging.Logger:
    """便捷函数，用于快速创建和配置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，可选值：debug, info, warning, error, critical
        log_dir: 日志文件存储目录
        detailed: 是否使用详细日志格式（包含文件名和行号）
    
    Returns:
        配置好的logging.Logger实例
    """
    config = {}
    
    if level:
        config["level"] = level
    
    if log_dir:
        config["log_dir"] = log_dir
    
    if detailed:
        config["format"] = DETAILED_LOG_FORMAT
    
    manager = LoggerManager(config)
    return manager.get_logger(name)


# 创建默认的日志管理器实例
default_manager = LoggerManager()

# 示例使用
if __name__ == "__main__":
    logger = setup_logger("test_logger", "debug", detailed=True)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

