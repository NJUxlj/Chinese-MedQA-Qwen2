"""
RAG服务
负责知识检索和上下文增强
"""

from typing import Dict, Any, List, Optional
import threading
import logging

from config.settings import settings
from providers.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class RAGService:
    """RAG服务类"""

    def __init__(self):
        """初始化RAG服务"""
        self._pipelines_lock = threading.RLock()

        cfg = settings.rag_service
        self.default_retriever_type = str(cfg.default_retriever)
        self._default_kb = str(cfg.default_kb) if cfg.default_kb else ""

        self.initialized = False

    def initialize(self) -> None:
        """初始化RAG服务"""
        if self.initialized:
            return
        logger.info("初始化RAG服务")
        self.initialized = True
        logger.info("RAG服务初始化成功")

    def _build_llm_provider(self, model_name: str = None) -> LLMProvider:
        """从 settings.llm 配置构建 LLMProvider"""
        cfg = settings.llm
        return LLMProvider(
            provider=str(cfg.model_provider),
            model_name=model_name or str(cfg.model_name),
            base_url=str(cfg.base_url) if hasattr(cfg, 'base_url') and cfg.base_url else None,
            api_key=str(cfg.api_key) if hasattr(cfg, 'api_key') and cfg.api_key else None,
            max_tokens=int(cfg.max_tokens) if hasattr(cfg, 'max_tokens') and cfg.max_tokens else 2048,
            temperature=float(cfg.temperature) if hasattr(cfg, 'temperature') and cfg.temperature else 0.7,
            top_p=float(cfg.top_p) if hasattr(cfg, 'top_p') and cfg.top_p else 0.9,
            timeout=int(cfg.timeout) if hasattr(cfg, 'timeout') and cfg.timeout else 60,
        )

    def generate_response(self, kb_name: str, query: str,
                          model_name: Optional[str] = None,
                          top_k: int = 5) -> Dict[str, Any]:
        """
        生成RAG增强响应（知识库功能暂时禁用，直接用LLM回答）

        Args:
            kb_name: 知识库名称
            query: 查询文本
            model_name: 模型名称
            top_k: 检索文档数量

        Returns:
            响应结果
        """
        model = self._build_llm_provider(model_name)

        prompt = f"""<|im_start|>system
你是一个专业的医疗助手，请基于可靠的医学知识回答用户的问题。
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        answer = model.generate(prompt=prompt)

        return {
            "answer": answer,
            "contexts": [],
            "sources": [],
        }

    def retrieve(self, kb_name: str, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """检索文档（暂时返回空列表）"""
        return []

    def get_available_knowledge_bases(self) -> List[Dict[str, Any]]:
        """获取可用知识库列表（暂时返回空）"""
        return []

    def create_knowledge_base(self, kb_name: str, description: str) -> bool:
        """创建知识库（暂时返回False）"""
        return False

    def delete_knowledge_base(self, kb_name: str) -> bool:
        """删除知识库（暂时返回False）"""
        return False

    def add_documents(self, kb_name: str, documents: List[Dict[str, Any]]) -> bool:
        """添加文档（暂时返回False）"""
        return False

# 单例模式
_rag_service = None
_lock = threading.Lock()

def get_rag_service() -> RAGService:
    """获取RAG服务单例"""
    global _rag_service
    if _rag_service is None:
        with _lock:
            if _rag_service is None:
                _rag_service = RAGService()
                _rag_service.initialize()
    return _rag_service
