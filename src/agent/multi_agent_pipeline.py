from typing import Dict, Any, Optional, List, Union, Tuple
import time
import uuid
import json

from .agent_base import AgentBase
from .agent_factory import AgentFactory
from .agent_manager import AgentManager
from ..utils.logger import get_logger
from ..models.base_model import BaseModel
from ..rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)

class MultiAgentPipeline:
    """
    多Agent协作流水线，用于复杂任务的多Agent协作处理
    """
    
    def __init__(
        self,
        model: BaseModel,
        rag_pipeline: Optional[RAGPipeline] = None,
        agent_manager: Optional[AgentManager] = None,
        pipeline_id: Optional[str] = None,
        name: str = "医疗多Agent系统",
        description: str = "一个由多个专业Agent组成的医疗问答系统",
        max_iterations: int = 10,
        verbose: bool = False
    ) -> None:
        """
        初始化多Agent流水线
        
        Args:
            model: 语言模型实例
            rag_pipeline: RAG流水线实例
            agent_manager: Agent管理器，如果不提供则创建新的
            pipeline_id: 流水线ID，如果不提供则自动生成
            name: 流水线名称
            description: 流水线描述
            max_iterations: 最大迭代次数
            verbose: 是否输出详细日志
        """
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.agent_manager = agent_manager or AgentManager()
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 对话历史
        self.conversation_history = []
        
        # 任务分配与协作配置
        self.task_router = None  # 用于任务路由的Agent
        self.coordinator = None  # 用于协调Agent的Agent
        self.specialized_agents = {}  # 专业Agent
        
        # 流水线状态
        self.is_initialized = False
    
    def _log(self, message: str) -> None:
        """
        记录日志
        
        Args:
            message: 日志消息
        """
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def initialize(self) -> None:
        """
        初始化多Agent流水线，创建任务路由器和协调员
        """
        if self.is_initialized:
            self._log("流水线已初始化")
            return
        
        try:
            # 创建任务路由器
            router_system_prompt = (
                "你是一个医疗任务路由器。你的职责是分析用户的医疗问题，确定最适合处理该问题的专业Agent。\n"
                "可用的专业Agent包括：\n"
                "1. 诊断Agent - 负责疾病诊断、症状分析\n"
                "2. 治疗Agent - 负责治疗方案、药物建议\n"
                "3. 预防Agent - 负责健康预防、生活方式建议\n"
                "4. 医学检查Agent - 负责医学检查、实验室结果解释\n"
                "\n"
                "对于每个用户查询，你应该输出最适合处理该问题的Agent名称，以及为什么选择该Agent的简短理由。"
            )
            
            router_agent_id = self.agent_manager.create_agent(
                agent_type="medical",
                model=self.model,
                name="任务路由器",
                description="负责将用户查询分配给适当的专业Agent",
                system_prompt=router_system_prompt,
                verbose=self.verbose
            )
            
            self.task_router = self.agent_manager.get_agent(router_agent_id)
            self._log(f"创建任务路由器: {router_agent_id}")
            
            # 创建协调员
            coordinator_system_prompt = (
                "你是一个医疗多Agent系统的协调员。你的职责是综合多个专业Agent的回答，提供最终的一致性响应。\n"
                "你应该解决不同Agent之间的冲突意见，确保最终回答是连贯的、完整的，并符合医学伦理规范。\n"
                "在综合意见时，你应该考虑每个Agent的专业领域，权衡不同意见的可靠性和适用性。\n"
                "最终回答应该简洁明了，专业但用户友好，避免过多的专业术语，除非必要。"
            )
            
            coordinator_agent_id = self.agent_manager.create_agent(
                agent_type="medical",
                model=self.model,
                name="协调员",
                description="负责整合多个专业Agent的回答，提供最终一致性响应",
                system_prompt=coordinator_system_prompt,
                verbose=self.verbose
            )
            
            self.coordinator = self.agent_manager.get_agent(coordinator_agent_id)
            self._log(f"创建协调员: {coordinator_agent_id}")
            
            # 创建专业Agent
            self._create_specialized_agents()
            
            self.is_initialized = True
            self._log("多Agent流水线初始化完成")
            
        except Exception as e:
            logger.error(f"初始化多Agent流水线失败: {str(e)}")
            raise
    
    def _create_specialized_agents(self) -> None:
        """创建专业Agent"""
        # 诊断Agent
        diagnosis_system_prompt = (
            "你是一个专注于医疗诊断的Agent。你的职责是分析用户描述的症状和体征，提供可能的诊断和鉴别诊断。\n"
            "你应该提出关键问题来完善诊断，并解释不同诊断的可能性和依据。\n"
            "在回答时要保持客观谨慎，避免确定性诊断，而是提供可能性分析和下一步建议。\n"
            "记住，你的建议不能替代专业医生的诊断，应当鼓励用户在需要时咨询医疗专业人员。"
        )
        
        diagnosis_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="诊断Agent",
            description="专注于疾病诊断和症状分析",
            system_prompt=diagnosis_system_prompt,
            verbose=self.verbose
        )
        
        self.specialized_agents["诊断Agent"] = self.agent_manager.get_agent(diagnosis_agent_id)
        self._log(f"创建诊断Agent: {diagnosis_agent_id}")
        
        # 治疗Agent
        treatment_system_prompt = (
            "你是一个专注于医疗治疗的Agent。你的职责是提供关于治疗方法、药物使用和治疗计划的信息。\n"
            "你应该基于用户的情况，提供关于常规治疗方案、药物选择、可能的副作用和治疗效果的信息。\n"
            "在回答时要平衡治疗的效果和风险，提供循证医学的证据支持，同时考虑治疗的个体化。\n"
            "记住，你的建议不能替代专业医生的处方，应当提醒用户遵医嘱，并在需要时咨询医疗专业人员。"
        )
        
        treatment_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="治疗Agent",
            description="专注于治疗方案和药物建议",
            system_prompt=treatment_system_prompt,
            verbose=self.verbose
        )
        
        self.specialized_agents["治疗Agent"] = self.agent_manager.get_agent(treatment_agent_id)
        self._log(f"创建治疗Agent: {treatment_agent_id}")
        
        # 预防Agent
        prevention_system_prompt = (
            "你是一个专注于疾病预防和健康维护的Agent。你的职责是提供关于健康生活方式、疾病预防和健康监测的信息。\n"
            "你应该基于用户的情况，提供关于饮食、运动、生活习惯和预防措施的建议，以减少疾病风险和提高生活质量。\n"
            "在回答时要强调预防的重要性，提供实用的健康建议，同时考虑用户的实际情况和可行性。\n"
            "记住，健康建议应当个体化，平衡理想与现实，鼓励用户逐步改善健康状况。"
        )
        
        prevention_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="预防Agent",
            description="专注于健康预防和生活方式建议",
            system_prompt=prevention_system_prompt,
            verbose=self.verbose
        )
        
        self.specialized_agents["预防Agent"] = self.agent_manager.get_agent(prevention_agent_id)
        self._log(f"创建预防Agent: {prevention_agent_id}")
        
        # 医学检查Agent
        examination_system_prompt = (
            "你是一个专注于医学检查和实验室结果解释的Agent。你的职责是提供关于各种医学检查、实验室检验的信息和结果解读。\n"
            "你应该解释各种检查的目的、过程、准备要求和结果含义，帮助用户理解检查结果的临床意义。\n"
            "在回答时要准确解读数据，解释正常值范围和异常值的潜在原因，同时避免过度解读或引起不必要的恐慌。\n"
            "记住，检查结果的解释应当结合临床背景，鼓励用户与医生讨论结果的具体含义和下一步措施。"
        )
        
        examination_agent_id = self.agent_manager.create_agent(
            agent_type="medical",
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="医学检查Agent",
            description="专注于医学检查和实验室结果解释",
            system_prompt=examination_system_prompt,
            verbose=self.verbose
        )
        
        self.specialized_agents["医学检查Agent"] = self.agent_manager.get_agent(examination_agent_id)
        self._log(f"创建医学检查Agent: {examination_agent_id}")
    
    def _route_task(self, query: str) -> List[str]:
        """
        根据用户查询路由任务到合适的专业Agent
        
        Args:
            query: 用户查询
            
        Returns:
            适合处理该查询的Agent名称列表
        """
        if not self.task_router:
            self._log("任务路由器未初始化")
            return list(self.specialized_agents.keys())  # 返回所有Agent
        
        routing_prompt = (
            f"请分析以下用户查询，确定最适合处理该问题的专业Agent。你的回答必须只包含Agent名称列表，格式为JSON数组，如[\"诊断Agent\", \"治疗Agent\"]。\n\n"
            f"用户查询: {query}\n\n"
            f"可用的专业Agent:\n"
            f"1. 诊断Agent - 负责疾病诊断、症状分析\n"
            f"2. 治疗Agent - 负责治疗方案、药物建议\n"
            f"3. 预防Agent - 负责健康预防、生活方式建议\n"
            f"4. 医学检查Agent - 负责医学检查、实验室结果解释"
        )
        
        routing_result = self.task_router.run(routing_prompt)
        response = routing_result.get("response", "")
        
        try:
            # 尝试从响应中提取JSON数组
            import re
            json_match = re.search(r'\[.*\]', response)
            
            if json_match:
                agent_names = json.loads(json_match.group())
                valid_agents = [name for name in agent_names if name in self.specialized_agents]
                
                if valid_agents:
                    self._log(f"路由任务到: {', '.join(valid_agents)}")
                    return valid_agents
            
            # 如果无法提取JSON或者列表为空，尝试提取Agent名称
            agent_names = []
            for name in self.specialized_agents.keys():
                if name in response:
                    agent_names.append(name)
            
            if agent_names:
                self._log(f"路由任务到: {', '.join(agent_names)}")
                return agent_names
            
            # 如果还是无法确定，返回所有Agent
            self._log("无法确定适合的Agent，使用所有Agent")
            return list(self.specialized_agents.keys())
            
        except Exception as e:
            logger.error(f"解析路由结果失败: {str(e)}")
            self._log("路由失败，使用所有Agent")
            return list(self.specialized_agents.keys())
    
    def _coordinate_responses(self, query: str, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        协调多个Agent的响应，生成最终一致的回答
        
        Args:
            query: 原始用户查询
            agent_responses: Agent响应字典 {Agent名称 -> 响应}
            
        Returns:
            最终协调后的响应
        """
        if not self.coordinator:
            self._log("协调员未初始化")
            
            # 如果只有一个Agent响应，直接返回
            if len(agent_responses) == 1:
                agent_name, response = list(agent_responses.items())[0]
                return response
            
            # 否则组合所有响应
            combined_response = "多Agent系统回答:\n\n"
            for agent_name, response_data in agent_responses.items():
                combined_response += f"【{agent_name}】: {response_data.get('response', '')}\n\n"
            
            return {
                "query": query,
                "response": combined_response,
                "metadata": {
                    "agent_responses": agent_responses,
                    "coordinated": False
                }
            }
        
        # 准备协调提示词
        coordination_prompt = (
            f"请协调以下多个专业Agent对用户查询的响应，提供一个连贯、一致的最终回答。解决冲突，确保回答全面且用户友好。\n\n"
            f"用户查询: {query}\n\n"
        )
        
        for agent_name, response_data in agent_responses.items():
            agent_response = response_data.get("response", "")
            coordination_prompt += f"【{agent_name}】响应:\n{agent_response}\n\n"
        
        coordination_prompt += "请整合以上专业Agent的回答，提供最终一致的响应。注意避免重复信息，解决冲突，确保回答连贯且全面。"
        
        # 获取协调后的响应
        result = self.coordinator.run(coordination_prompt)
        
        # 添加元数据
        result["metadata"]["agent_responses"] = agent_responses
        result["metadata"]["coordinated"] = True
        
        return result
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行多Agent流水线处理用户查询
        
        Args:
            query: 用户查询
            kwargs: 其他参数
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 确保流水线已初始化
        if not self.is_initialized:
            self.initialize()
        
        # 记录用户查询
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # 路由任务到适合的Agent
        relevant_agents = self._route_task(query)
        
        # 并行查询相关Agent
        agent_responses = {}
        for agent_name in relevant_agents:
            if agent_name in self.specialized_agents:
                agent = self.specialized_agents[agent_name]
                self._log(f"查询 {agent_name}...")
                try:
                    response = agent.run(query, **kwargs)
                    agent_responses[agent_name] = response
                except Exception as e:
                    logger.error(f"{agent_name} 处理查询失败: {str(e)}")
                    agent_responses[agent_name] = {
                        "query": query,
                        "response": f"处理失败: {str(e)}",
                        "metadata": {"error": str(e)}
                    }
        
        # 协调多个Agent的响应
        final_response = self._coordinate_responses(query, agent_responses)
        
        # 记录系统响应
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response.get("response", "")
        })
        
        # 添加执行时间到元数据
        if "metadata" not in final_response:
            final_response["metadata"] = {}
        final_response["metadata"]["execution_time"] = time.time() - start_time
        final_response["metadata"]["pipeline_id"] = self.pipeline_id
        final_response["metadata"]["relevant_agents"] = relevant_agents
        
        return final_response
    
    def reset(self) -> None:
        """重置多Agent流水线状态"""
        # 清空对话历史
        self.conversation_history = []
        
        # 重置所有Agent
        if self.task_router:
            self.task_router.reset()
        
        if self.coordinator:
            self.coordinator.reset()
        
        for agent in self.specialized_agents.values():
            agent.reset()
        
        self._log("多Agent流水线已重置")