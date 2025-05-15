from typing import Dict, Any, Optional, List, Union
import uuid

from .agent_base import AgentBase
from .agent_factory import AgentFactory
from ..utils.logger import get_logger
from ..config.agent_config import AgentConfig
from ..models.base_model import BaseModel
from ..rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)

class AgentManager:
    """
    Agent管理器，用于管理多个Agent实例
    """
    
    def __init__(self) -> None:
        """初始化Agent管理器"""
        self.agents = {}  # agent_id -> agent实例
    
    def create_agent(
        self,
        agent_type: str,
        model: BaseModel,
        rag_pipeline: Optional[RAGPipeline] = None,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        创建新Agent实例
        
        Args:
            agent_type: Agent类型
            model: 语言模型实例
            rag_pipeline: RAG流水线实例
            config: Agent配置
            agent_id: 指定Agent ID，如果不提供则自动生成
            kwargs: 其他参数
            
        Returns:
            创建的Agent ID
        """
        # 如果没有提供ID，生成一个
        if not agent_id:
            agent_id = str(uuid.uuid4())
        
        # 如果ID已存在，生成一个新的
        if agent_id in self.agents:
            logger.warning(f"Agent ID '{agent_id}' 已存在，生成新ID")
            agent_id = str(uuid.uuid4())
        
        # 创建Agent
        try:
            agent = AgentFactory.create_agent(
                agent_type=agent_type,
                model=model,
                rag_pipeline=rag_pipeline,
                config=config,
                agent_id=agent_id,
                **kwargs
            )
            
            self.agents[agent_id] = agent
            logger.info(f"创建Agent '{agent_id}' 成功，类型: {agent_type}")
            return agent_id
            
        except Exception as e:
            logger.error(f"创建Agent失败: {str(e)}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[AgentBase]:
        """
        获取Agent实例
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent实例，如果不存在则返回None
        """
        return self.agents.get(agent_id)
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        移除Agent实例
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功移除
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"移除Agent '{agent_id}' 成功")
            return True
        
        logger.warning(f"尝试移除不存在的Agent '{agent_id}'")
        return False
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        列出所有Agent
        
        Returns:
            Agent信息列表
        """
        return [
            {
                "id": agent_id,
                "name": agent.name,
                "description": agent.description,
                "model": agent.model_name if hasattr(agent, 'model_name') else None
            }
            for agent_id, agent in self.agents.items()
        ]
    
    def query_agent(self, agent_id: str, query: str, **kwargs) -> Dict[str, Any]:
        """
        向指定Agent发送查询
        
        Args:
            agent_id: Agent ID
            query: 用户查询
            kwargs: 其他参数
            
        Returns:
            Agent响应
            
        Raises:
            ValueError: 如果找不到Agent
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"找不到Agent '{agent_id}'")
        
        try:
            result = agent.run(query, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Agent '{agent_id}' 处理查询失败: {str(e)}")
            raise