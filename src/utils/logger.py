
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
    """日志管理器，处理日志记录器的创建和配置"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _config: Dict[str, Any] = {}
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化日志管理器"""
        if self._initialized:
            return
        
        self._config = DEFAULT_LOG_CONFIG.copy()
        if config:
            self._config.update(config)
        
        # 确保日志目录存在
        if self._config["file_output"]:
            os.makedirs(self._config["log_dir"], exist_ok=True)
        
        self._initialized = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的logger"""
        if name in self._loggers:
            return self._loggers[name]
        
        # 创建新的logger
        logger = logging.getLogger(name)
        
        # 设置日志级别
        level = LOG_LEVELS.get(self._config["level"].lower(), logging.INFO)
        logger.setLevel(level)
        
        # 防止日志重复
        logger.propagate = False
        
        # 清除已有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 日志格式
        log_format = self._config.get("format", DEFAULT_LOG_FORMAT)
        formatter = logging.Formatter(log_format)
        
        # 控制台输出
        if self._config["console_output"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件输出
        if self._config["file_output"]:
            log_file = os.path.join(
                self._config["log_dir"], 
                f"{name}_{time.strftime('%Y%m%d')}.log"
            )
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self._config["max_bytes"],
                backupCount=self._config["backup_count"]
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 保存logger实例
        self._loggers[name] = logger
        return logger
    
    def set_level(self, level: str):
        """设置全局日志级别"""
        if level.lower() not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}")
        
        self._config["level"] = level.lower()
        level_value = LOG_LEVELS[level.lower()]
        
        for logger in self._loggers.values():
            logger.setLevel(level_value)
    
    def update_config(self, config: Dict[str, Any]):
        """更新日志配置"""
        self._config.update(config)
        
        # 重置所有logger
        for name in list(self._loggers.keys()):
            self._loggers.pop(name)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    detailed: bool = False
) -> logging.Logger:
    """便捷函数，用于设置和获取日志记录器"""
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

