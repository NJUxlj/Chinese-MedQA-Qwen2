from typing import Dict, Any, Optional, List, Union
import importlib

from .agent_base import AgentBase
from .medical_agent import MedicalAgent
from .tool_manager import ToolManager
from .tools.tool_base import ToolBase
from .tools.search_tool import SearchTool, BingSearchTool
from .tools.calculator_tool import CalculatorTool
from .tools.medical_reference_tool import MedicalReferenceTool
from .tools.reaction_agent_tool import ReActAgentTool
from .tools.medical_assessment_tool import MedicalAssessmentTool

from ..utils.logger import get_logger
from ..config.agent_config import AgentConfig
from ..models.base_model import BaseModel
from ..rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)

class AgentFactory:
    """
    Agent工厂类，用于创建和配置不同类型的Agent
    """
    
    @staticmethod
    def create_agent(
        agent_type: str,
        model: BaseModel,
        rag_pipeline: Optional[RAGPipeline] = None,
        config: Optional[AgentConfig] = None,
        **kwargs
    ) -> AgentBase:
        """
        创建Agent实例
        
        Args:
            agent_type: Agent类型，如'medical'
            model: 语言模型实例
            rag_pipeline: RAG流水线实例，用于检索相关医疗知识
            config: Agent配置
            kwargs: 其他参数
            
        Returns:
            Agent实例
            
        Raises:
            ValueError: 如果Agent类型不支持
        """
        if agent_type.lower() == 'medical':
            agent = MedicalAgent(
                model=model,
                rag_pipeline=rag_pipeline,
                **kwargs
            )
            
            # 如果有配置，加载默认工具
            if config and config.load_default_tools:
                AgentFactory.load_default_tools(agent, model, config)
            
            return agent
        else:
            raise ValueError(f"不支持的Agent类型: {agent_type}")
    
    @staticmethod
    def load_default_tools(
        agent: AgentBase,
        model: BaseModel,
        config: AgentConfig
    ) -> None:
        """
        加载默认工具到Agent
        
        Args:
            agent: Agent实例
            model: 语言模型实例
            config: Agent配置
        """
        # 加载计算工具
        if config.enable_calculator_tool:
            calculator_tool = CalculatorTool()
            agent.add_tool(calculator_tool)
            logger.info("已加载计算工具")
        
        # 加载搜索工具
        if config.enable_search_tool:
            if config.search_api_key and config.search_engine_id:
                search_tool = SearchTool(
                    search_api_key=config.search_api_key,
                    search_engine_id=config.search_engine_id
                )
                agent.add_tool(search_tool)
                logger.info("已加载Google搜索工具")
            
            if config.bing_subscription_key:
                bing_search_tool = BingSearchTool(
                    subscription_key=config.bing_subscription_key
                )
                agent.add_tool(bing_search_tool)
                logger.info("已加载Bing搜索工具")
        
        # 加载医疗参考工具
        if config.enable_medical_reference_tool:
            medical_reference_tool = MedicalReferenceTool(
                reference_data_path=config.medical_reference_data_path
            )
            agent.add_tool(medical_reference_tool)
            logger.info("已加载医疗参考工具")
        
        # 加载医疗评估工具
        if config.enable_medical_assessment_tool:
            medical_assessment_tool = MedicalAssessmentTool()
            agent.add_tool(medical_assessment_tool)
            logger.info("已加载医疗评估工具")
        
        # 加载ReAct Agent工具
        if config.enable_react_agent_tool:
            # 创建一个工具管理器，包含已加载的所有工具
            tool_manager = agent.tool_manager if hasattr(agent, 'tool_manager') else ToolManager()
            available_tools = tool_manager.get_all_tools()
            
            react_agent_tool = ReActAgentTool(
                model=model,
                available_tools=available_tools
            )
            agent.add_tool(react_agent_tool)
            logger.info("已加载ReAct Agent工具")
    
    @staticmethod
    def load_custom_tool(
        agent: AgentBase,
        tool_module_path: str,
        tool_class_name: str,
        **tool_kwargs
    ) -> bool:
        """
        加载自定义工具到Agent
        
        Args:
            agent: Agent实例
            tool_module_path: 工具模块路径
            tool_class_name: 工具类名
            tool_kwargs: 传递给工具构造函数的参数
            
        Returns:
            是否成功加载
            
        Raises:
            ImportError: 如果无法导入工具模块
            AttributeError: 如果找不到工具类
        """
        try:
            module = importlib.import_module(tool_module_path)
            tool_class = getattr(module, tool_class_name)
            
            if not issubclass(tool_class, ToolBase):
                logger.error(f"工具类 {tool_class_name} 不是ToolBase的子类")
                return False
            
            tool_instance = tool_class(**tool_kwargs)
            agent.add_tool(tool_instance)
            logger.info(f"已加载自定义工具 {tool_class_name}")
            return True
            
        except ImportError as e:
            logger.error(f"导入工具模块 {tool_module_path} 失败: {str(e)}")
            return False
        except AttributeError as e:
            logger.error(f"找不到工具类 {tool_class_name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"加载自定义工具失败: {str(e)}")
            return False