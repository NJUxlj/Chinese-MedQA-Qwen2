"""
MDAgents系统测试套件
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdagents.config import (
    SystemConfig, LLMConfig, AgentConfig, 
    ComplexityLevel, CollaborationPattern,
    get_config_manager, ConfigManager
)
from mdagents.core.complexity_analyzer import ComplexityAnalyzer
from mdagents.core.collaboration_allocator import CollaborationAllocator
from mdagents.core.consensus_mechanism import ConsensusMechanism, ConsensusResult
from mdagents.core.main_controller import MainController


class TestConfig(unittest.TestCase):
    """测试配置模块"""
    
    def test_llm_config_creation(self):
        """测试LLM配置创建"""
        config = LLMConfig(
            model_name="qwen2",
            max_tokens=2048,
            temperature=0.5
        )
        
        self.assertEqual(config.model_name, "qwen2")
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.temperature, 0.5)
        
        # 测试转换为字典
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model_name"], "qwen2")
    
    def test_agent_config_creation(self):
        """测试智能体配置创建"""
        config = AgentConfig(
            name="测试智能体",
            agent_type="pcc",
            description="测试用全科医生智能体",
            capabilities=["基础诊断", "健康建议"]
        )
        
        self.assertEqual(config.name, "测试智能体")
        self.assertEqual(config.agent_type, "pcc")
        self.assertIn("基础诊断", config.capabilities)
    
    def test_system_config_creation(self):
        """测试系统配置创建"""
        llm_config = LLMConfig(model_name="test-model")
        
        config = SystemConfig(llm=llm_config)
        
        self.assertEqual(config.llm.model_name, "test-model")
        self.assertTrue(config.enable_async_processing)
    
    def test_config_manager(self):
        """测试配置管理器"""
        manager = ConfigManager()
        
        # 测试获取配置
        config = manager.config
        self.assertIsInstance(config, SystemConfig)
        
        # 测试更新配置
        manager.update_config({"max_concurrent_sessions": 5})
        self.assertEqual(manager.config.max_concurrent_sessions, 5)
    
    def test_config_validation(self):
        """测试配置验证"""
        manager = ConfigManager()
        
        # 测试无效配置
        errors = manager.validate_config()
        # 默认配置应该通过验证
        self.assertEqual(len(errors), 0)


class TestComplexityAnalyzer(unittest.TestCase):
    """测试复杂度分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = ComplexityAnalyzer()
    
    def test_analyze_simple_query(self):
        """测试简单查询分析"""
        query = "我有点头痛，需要怎么办？"
        patient_info = {"age": 25, "gender": "female"}
        
        result = self.analyzer.analyze_complexity(query, patient_info)
        
        self.assertIn("complexity_level", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["complexity_level"], str)
        self.assertIsInstance(result["confidence"], float)
    
    def test_analyze_complex_query(self):
        """测试复杂查询分析"""
        query = "患者出现胸痛、呼吸困难，伴有发热，血压升高，需要紧急处理"
        patient_info = {"age": 65, "gender": "male", "history": ["hypertension", "diabetes"]}
        
        result = self.analyzer.analyze_complexity(query, patient_info)
        
        # 复杂查询应该被识别为高复杂度
        self.assertIn("complexity_level", result)
        self.assertIn("details", result)
    
    def test_analyze_without_patient_info(self):
        """测试无患者信息的分析"""
        query = "普通感冒症状"
        
        result = self.analyzer.analyze_complexity(query)
        
        self.assertIn("complexity_level", result)
        self.assertIsInstance(result["confidence"], float)


class TestCollaborationAllocator(unittest.TestCase):
    """测试协作分配器"""
    
    def setUp(self):
        """设置测试环境"""
        self.allocator = CollaborationAllocator()
    
    def test_allocate_for_simple_case(self):
        """测试简单病例的协作分配"""
        complexity_level = ComplexityLevel.SIMPLE
        query = "轻微头痛"
        
        result = self.allocator.allocate_collaboration(complexity_level, query)
        
        self.assertIn("pattern", result)
        self.assertIn("agents_needed", result)
        self.assertIsInstance(result["agents_needed"], list)
    
    def test_allocate_for_complex_case(self):
        """测试复杂病例的协作分配"""
        complexity_level = ComplexityLevel.COMPLEX
        query = "多系统疾病，需要专家会诊"
        
        result = self.allocator.allocate_collaboration(complexity_level, query)
        
        # 复杂病例应该需要更多专家
        self.assertIn("pattern", result)
        self.assertGreaterEqual(len(result["agents_needed"]), 2)


class TestConsensusMechanism(unittest.TestCase):
    """测试共识机制"""
    
    def setUp(self):
        """设置测试环境"""
        self.consensus = ConsensusMechanism()
    
    def test_simple_majority_consensus(self):
        """测试简单多数共识"""
        expert_opinions = [
            {
                "agent_name": "专家A",
                "agent_type": "pcc",
                "diagnosis": "感冒",
                "confidence": 0.8,
                "recommendations": ["多休息", "多喝水"]
            },
            {
                "agent_name": "专家B", 
                "agent_type": "specialist",
                "diagnosis": "感冒",
                "confidence": 0.9,
                "recommendations": ["休息", "对症治疗"]
            },
            {
                "agent_name": "专家C",
                "agent_type": "pcc", 
                "diagnosis": "流感",
                "confidence": 0.7,
                "recommendations": ["抗病毒治疗"]
            }
        ]
        
        result = self.consensus.achieve_consensus(
            expert_opinions=expert_opinions,
            algorithm="simple_majority",
            decision_type="diagnosis"
        )
        
        self.assertIsInstance(result, ConsensusResult)
        self.assertIn("final_decision", result.final_decision)
        self.assertIsInstance(result.success, bool)
    
    def test_weighted_voting_consensus(self):
        """测试加权投票共识"""
        expert_opinions = [
            {
                "agent_name": "专科医生",
                "agent_type": "specialist",
                "diagnosis": "肺炎",
                "confidence": 0.9,
                "recommendations": ["抗生素治疗"]
            },
            {
                "agent_name": "全科医生",
                "agent_type": "pcc",
                "diagnosis": "支气管炎", 
                "confidence": 0.6,
                "recommendations": ["对症治疗"]
            }
        ]
        
        result = self.consensus.achieve_consensus(
            expert_opinions=expert_opinions,
            algorithm="weighted_voting",
            decision_type="diagnosis"
        )
        
        self.assertIsInstance(result, ConsensusResult)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestMainController(unittest.TestCase):
    """测试主控制器"""
    
    def setUp(self):
        """设置测试环境"""
        # 使用模拟的LLM配置
        self.controller = MainController(llm_config=None)
    
    def test_controller_initialization(self):
        """测试控制器初始化"""
        self.assertEqual(self.controller.system_status, "initialized")
        self.assertIsInstance(self.controller.agents, dict)
        self.assertGreater(len(self.controller.agents), 0)
    
    def test_system_status(self):
        """测试系统状态获取"""
        status = self.controller.get_system_status()
        
        self.assertIn("system_status", status)
        self.assertIn("total_decisions", status)
        self.assertIn("active_agents", status)
        self.assertEqual(status["system_status"], "initialized")
    
    def test_add_custom_agent(self):
        """测试添加自定义智能体"""
        from mdagents.agents.base_agent import BaseAgent
        from mdagents.config import AgentConfig
        
        # 创建模拟智能体
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.agent_type = "custom"
        mock_agent.name = "自定义智能体"
        
        # 添加自定义智能体
        self.controller.add_custom_agent(mock_agent)
        
        self.assertIn("custom", self.controller.agents)
        self.assertEqual(self.controller.agents["custom"].name, "自定义智能体")
    
    @patch('asyncio.sleep')
    async def test_process_medical_query(self, mock_sleep):
        """测试医疗查询处理（异步测试）"""
        query = "患者感到头痛和发热"
        patient_info = {"age": 30, "gender": "female"}
        
        # 使用asyncio运行异步测试
        result = await self.controller.process_medical_query(query, patient_info)
        
        self.assertIn("session_id", result)
        self.assertIn("status", result)
        self.assertIn("complexity_analysis", result)
        self.assertIn("collaboration_plan", result)
        self.assertIn("consensus_result", result)
        self.assertIn("final_report", result)
        
        # 验证异步调用
        mock_sleep.assert_called()
    
    def test_remove_agent(self):
        """测试移除智能体"""
        # 先确保智能体存在
        initial_count = len(self.controller.agents)
        
        # 移除一个智能体
        self.controller.remove_agent("pcc")
        
        self.assertEqual(len(self.controller.agents), initial_count - 1)
        self.assertNotIn("pcc", self.controller.agents)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    async def test_full_workflow(self):
        """测试完整工作流程"""
        controller = MainController(llm_config=None)
        
        # 模拟一个完整的医疗查询流程
        query = "患者出现胸痛和呼吸急促"
        patient_info = {
            "age": 55,
            "gender": "male", 
            "history": ["hypertension"],
            "symptoms": ["chest_pain", "shortness_of_breath"]
        }
        
        result = await controller.process_medical_query(query, patient_info)
        
        # 验证完整流程的各个环节
        self.assertEqual(result["status"], "success")
        self.assertIn("complexity_analysis", result)
        self.assertIn("collaboration_plan", result)
        self.assertIn("expert_agents", result)
        self.assertIn("consensus_result", result)
        self.assertIn("final_report", result)
        
        # 验证系统状态更新
        status = controller.get_system_status()
        self.assertEqual(status["total_decisions"], 1)


def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestConfig,
        TestComplexityAnalyzer, 
        TestCollaborationAllocator,
        TestConsensusMechanism,
        TestMainController,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == "__main__":
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print(f"\n测试完成!")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # 设置退出码
    exit(0 if result.wasSuccessful() else 1)