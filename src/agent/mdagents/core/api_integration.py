"""
API集成模块
负责集成各种API调用功能，支持不同的大语言模型
"""

from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from models.api_model import ApiModel, ZhipuApiModel, OpenAIApiModel
from config.settings import settings
from ..tools.logger import setup_logger


class ApiIntegrationManager:
    """API集成管理器"""
    
    def __init__(self, llm_config=None):
        """
        初始化API集成管理器

        Args:
            llm_config: LLM配置对象
        """
        self.llm_config = llm_config if llm_config is not None else settings.llm
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化API模型
        self.api_model = self._initialize_api_model()
        
        # API调用历史
        self.api_call_history = []
        
        # 提示词模板
        self.prompt_templates = self._initialize_prompt_templates()
        
        # API调用统计
        self.call_statistics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0.0
        }
    
    def _initialize_api_model(self) -> ApiModel:
        """初始化API模型"""
        try:
            if self.llm_config.model_provider == "zhipuai":
                return ZhipuApiModel(
                    api_key=self.llm_config.api_key,
                    api_base=self.llm_config.base_url,
                    model_name=self.llm_config.model_name
                )
            elif self.llm_config.model_provider == "openai":
                return OpenAIApiModel(
                    api_key=self.llm_config.api_key,
                    api_base=self.llm_config.base_url,
                    model_name=self.llm_config.model_name
                )
            else:
                raise ValueError(f"不支持的模型提供商: {self.llm_config.model_provider}")
                
        except Exception as e:
            self.logger.error(f"API模型初始化失败: {str(e)}")
            raise
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """初始化提示词模板"""
        return {
            "medical_analysis": """
你是一位专业的医疗AI助手，具备丰富的医学知识和临床经验。请基于以下信息进行分析：

患者信息：{patient_info}

查询内容：{query}

请提供：
1. 初步诊断分析
2. 推荐治疗方案
3. 必要的检查建议
4. 用药指导（如适用）
5. 风险评估
6. 随访建议

请确保建议科学、准确、安全。
""",
            
            "specialist_consultation": """
你是一位{specialty}专家，具有该领域的深度专业知识。请针对以下医疗问题提供专业意见：

患者信息：{patient_info}

具体问题：{query}

专业建议：
1. 诊断意见
2. 治疗建议
3. 检查推荐
4. 用药方案
5. 注意事项

请基于循证医学原则和临床指南提供建议。
""",
            
            "complexity_assessment": """
请分析以下医疗查询的复杂程度：

查询内容：{query}

患者背景：{patient_context}

复杂度评估：
1. 医学复杂度等级（低/中/高）
2. 需要的专科数量
3. 风险评估
4. 推荐协作模式
5. 决策难度

请提供详细的分析理由。
""",
            
            "team_recruitment": """
基于以下查询招募合适的医疗专家团队：

查询内容：{query}
复杂度等级：{complexity_level}
患者情况：{patient_context}

请推荐：
1. 需要哪些专科专家
2. 专家数量
3. 团队构成理由
4. 协作模式
5. 预期处理时间

考虑医疗质量和效率的平衡。
""",
            
            "quality_review": """
请对以下医疗建议进行质量审查：

原始查询：{query}
医疗建议：{medical_advice}
专家建议：{expert_recommendations}

审查维度：
1. 医学准确性
2. 安全性评估
3. 循证依据
4. 完整性检查
5. 沟通效果

请提供详细的审查结果和改进建议。
""",
            
            "consensus_integration": """
请整合多位专家的建议，形成统一的医疗决策：

查询内容：{query}
专家建议：{expert_recommendations}
专家权重：{expert_weights}

整合要求：
1. 分析专家意见的一致性和分歧
2. 选择最优的诊断和治疗方案
3. 评估整体信心水平
4. 识别需要进一步确认的问题
5. 生成最终的综合建议

请提供详细的整合过程和结果。
"""
        }
    
    def call_api(self, prompt_type: str, **kwargs) -> Dict[str, Any]:
        """
        调用API
        
        Args:
            prompt_type: 提示词类型
            **kwargs: 提示词参数
            
        Returns:
            API调用结果
        """
        start_time = datetime.now()
        
        try:
            # 获取提示词模板
            if prompt_type not in self.prompt_templates:
                raise ValueError(f"未知的提示词类型: {prompt_type}")
            
            template = self.prompt_templates[prompt_type]
            
            # 格式化提示词
            prompt = template.format(**kwargs)
            
            self.logger.info(f"调用{prompt_type}类型的API")
            
            # 调用API
            response = self.api_model.generate(
                prompt=prompt,
                additional_args={
                    "temperature": self.llm_config.temperature,
                    "top_p": self.llm_config.top_p,
                    "max_tokens": self.llm_config.max_tokens
                }
            )
            
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录成功调用
            self._record_api_call(
                prompt_type=prompt_type,
                prompt=prompt,
                response=response,
                response_time=response_time,
                status="success"
            )
            
            self.logger.info(f"API调用成功，响应时间: {response_time:.2f}秒")
            
            return {
                "status": "success",
                "response": response,
                "prompt_type": prompt_type,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录失败调用
            self._record_api_call(
                prompt_type=prompt_type,
                prompt=kwargs.get("query", ""),
                response=None,
                response_time=response_time,
                status="failed",
                error=str(e)
            )
            
            self.logger.error(f"API调用失败: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "prompt_type": prompt_type,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
    
    def _record_api_call(self, prompt_type: str, prompt: str, response: str, 
                        response_time: float, status: str, error: str = None):
        """记录API调用"""
        call_record = {
            "prompt_type": prompt_type,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,  # 截取前200字符
            "response": response[:500] + "..." if response and len(response) > 500 else response,
            "response_time": response_time,
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.api_call_history.append(call_record)
        
        # 更新统计信息
        self.call_statistics["total_calls"] += 1
        if status == "success":
            self.call_statistics["successful_calls"] += 1
        else:
            self.call_statistics["failed_calls"] += 1
        
        # 计算平均响应时间
        successful_calls = [call for call in self.api_call_history if call["status"] == "success"]
        if successful_calls:
            total_time = sum(call["response_time"] for call in successful_calls)
            self.call_statistics["average_response_time"] = total_time / len(successful_calls)
        
        # 保持历史记录在合理范围内
        if len(self.api_call_history) > 1000:
            self.api_call_history = self.api_call_history[-500:]
    
    def call_api_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        使用消息格式调用API
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            API调用结果
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("调用API（消息格式）")
            
            # 调用API
            response = self.api_model.generate(
                prompt=None,
                messages=messages,
                additional_args={
                    "temperature": self.llm_config.temperature,
                    "top_p": self.llm_config.top_p,
                    "max_tokens": self.llm_config.max_tokens
                }
            )
            
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录成功调用
            self._record_api_call(
                prompt_type="custom_messages",
                prompt=json.dumps(messages)[:200],
                response=response,
                response_time=response_time,
                status="success"
            )
            
            self.logger.info(f"API调用成功，响应时间: {response_time:.2f}秒")
            
            return {
                "status": "success",
                "response": response,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录失败调用
            self._record_api_call(
                prompt_type="custom_messages",
                prompt=json.dumps(messages)[:200],
                response=None,
                response_time=response_time,
                status="failed",
                error=str(e)
            )
            
            self.logger.error(f"API调用失败: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        获取文本嵌入
        
        Args:
            texts: 文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            嵌入结果
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"获取{len(texts) if isinstance(texts, list) else 1}个文本的嵌入")
            
            # 调用API获取嵌入
            embeddings = self.api_model.get_embeddings(texts, **kwargs)
            
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录调用
            self._record_api_call(
                prompt_type="embeddings",
                prompt=str(texts)[:200],
                response=str(embeddings.shape),
                response_time=response_time,
                status="success"
            )
            
            self.logger.info(f"嵌入获取成功，响应时间: {response_time:.2f}秒")
            
            return {
                "status": "success",
                "embeddings": embeddings,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 记录失败调用
            self._record_api_call(
                prompt_type="embeddings",
                prompt=str(texts)[:200],
                response=None,
                response_time=response_time,
                status="failed",
                error=str(e)
            )
            
            self.logger.error(f"嵌入获取失败: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
    
    def batch_call_api(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量API调用
        
        Args:
            requests: 请求列表，每个请求包含prompt_type和参数
            
        Returns:
            批量调用结果
        """
        results = []
        
        self.logger.info(f"开始批量API调用，共{len(requests)}个请求")
        
        for i, request in enumerate(requests):
            try:
                prompt_type = request.get("prompt_type")
                prompt_args = request.get("kwargs", {})
                
                result = self.call_api(prompt_type, **prompt_args)
                result["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"批量调用第{i}个请求失败: {str(e)}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "batch_index": i,
                    "timestamp": datetime.now().isoformat()
                })
        
        self.logger.info(f"批量API调用完成，成功{sum(1 for r in results if r['status'] == 'success')}个")
        
        return results
    
    def create_medical_messages(self, role: str, content: str, 
                              system_prompt: str = None) -> List[Dict[str, str]]:
        """
        创建医疗对话消息
        
        Args:
            role: 角色（user/assistant/system）
            content: 内容
            system_prompt: 系统提示词
            
        Returns:
            消息列表
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": role,
            "content": content
        })
        
        return messages
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        recent_calls = self.api_call_history[-100:]  # 最近100次调用
        
        # 按类型统计
        type_stats = {}
        for call in recent_calls:
            prompt_type = call["prompt_type"]
            if prompt_type not in type_stats:
                type_stats[prompt_type] = {"total": 0, "success": 0, "failed": 0}
            type_stats[prompt_type]["total"] += 1
            if call["status"] == "success":
                type_stats[prompt_type]["success"] += 1
            else:
                type_stats[prompt_type]["failed"] += 1
        
        # 响应时间统计
        successful_calls = [call for call in recent_calls if call["status"] == "success"]
        response_times = [call["response_time"] for call in successful_calls]
        
        return {
            "overall_statistics": self.call_statistics,
            "recent_calls_count": len(recent_calls),
            "type_statistics": type_stats,
            "response_time_stats": {
                "average": sum(response_times) / len(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "median": sorted(response_times)[len(response_times)//2] if response_times else 0
            },
            "success_rate": (
                self.call_statistics["successful_calls"] / max(self.call_statistics["total_calls"], 1)
            ) * 100
        }
    
    def clear_history(self):
        """清空历史记录"""
        self.api_call_history.clear()
        self.call_statistics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0.0
        }
        self.logger.info("API调用历史已清空")


class ApiModelWrapper:
    """API模型包装器"""
    
    def __init__(self, api_integration_manager: ApiIntegrationManager):
        """
        初始化API模型包装器
        
        Args:
            api_integration_manager: API集成管理器
        """
        self.api_manager = api_integration_manager
        self.logger = setup_logger(self.__class__.__name__)
    
    def generate_medical_analysis(self, query: str, patient_info: str = "") -> Dict[str, Any]:
        """生成医疗分析"""
        return self.api_manager.call_api(
            prompt_type="medical_analysis",
            query=query,
            patient_info=patient_info
        )
    
    def generate_specialist_consultation(self, query: str, specialty: str, 
                                       patient_info: str = "") -> Dict[str, Any]:
        """生成专科咨询"""
        return self.api_manager.call_api(
            prompt_type="specialist_consultation",
            query=query,
            specialty=specialty,
            patient_info=patient_info
        )
    
    def assess_complexity(self, query: str, patient_context: str = "") -> Dict[str, Any]:
        """评估复杂度"""
        return self.api_manager.call_api(
            prompt_type="complexity_assessment",
            query=query,
            patient_context=patient_context
        )
    
    def recruit_team(self, query: str, complexity_level: str, 
                    patient_context: str = "") -> Dict[str, Any]:
        """招募团队"""
        return self.api_manager.call_api(
            prompt_type="team_recruitment",
            query=query,
            complexity_level=complexity_level,
            patient_context=patient_context
        )
    
    def review_quality(self, query: str, medical_advice: str, 
                      expert_recommendations: str) -> Dict[str, Any]:
        """质量审查"""
        return self.api_manager.call_api(
            prompt_type="quality_review",
            query=query,
            medical_advice=medical_advice,
            expert_recommendations=expert_recommendations
        )
    
    def integrate_consensus(self, query: str, expert_recommendations: str, 
                          expert_weights: str = "") -> Dict[str, Any]:
        """共识整合"""
        return self.api_manager.call_api(
            prompt_type="consensus_integration",
            query=query,
            expert_recommendations=expert_recommendations,
            expert_weights=expert_weights
        )