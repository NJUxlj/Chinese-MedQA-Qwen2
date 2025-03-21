
import os
import sys
import logging
import json
import yaml
from typing import Dict, List, Any, Optional

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    设置日志记录。
    
    Args:
        log_file: 日志文件路径（可选）
        level: 日志级别
    """
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果提供）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return root_logger

def load_config(config_path: str) -> Dict:
    """
    加载YAML配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"加载配置文件 {config_path} 时出错: {e}")

def format_pubmed_results(results: List[Dict]) -> str:
    """
    格式化PubMed搜索结果以便于显示。
    
    Args:
        results: PubMed搜索结果列表
        
    Returns:
        格式化的结果字符串
    """
    if not results:
        return "未找到结果。"
        
    formatted = ""
    for i, article in enumerate(results, 1):
        formatted += f"{i}. {article.get('title')}\n"
        formatted += f"   作者: {article.get('authors')}\n"
        formatted += f"   发表: {article.get('journal')} ({article.get('publication_date')})\n"
        formatted += f"   PMID: {article.get('pmid')}\n"
        formatted += f"   链接: {article.get('pubmed_link')}\n"
        formatted += f"   摘要: {article.get('abstract')[:200]}...\n\n"
        
    return formatted

def save_conversation(conversation: List[Dict], file_path: str):
    """
    保存对话历史到JSON文件。
    
    Args:
        conversation: 对话历史列表
        file_path: 保存文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存对话历史时出错: {e}")

def load_conversation(file_path: str) -> List[Dict]:
    """
    从JSON文件加载对话历史。
    
    Args:
        file_path: 对话历史文件路径
        
    Returns:
        对话历史列表
    """
    if not os.path.exists(file_path):
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载对话历史时出错: {e}")
        return []
