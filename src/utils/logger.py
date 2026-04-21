
# utils/logger.py

import os
import sys
import numbers
import pprint
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Optional, Dict, Any
import torch
from datetime import datetime

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

# 默认日志目录：指向项目根目录下的 logs/ 文件夹（即 <project_root>/logs）
# __file__ = src/utils/logger.py → 上溯三级得到 <project_root>
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
DEFAULT_LOG_CONFIG = {
    "level": "info",
    "format": DEFAULT_LOG_FORMAT,
    "log_dir": DEFAULT_LOG_DIR,
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





class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")








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

