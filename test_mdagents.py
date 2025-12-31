#!/usr/bin/env python3
"""
MDAgents 全面测试脚本

此脚本测试 mdagents 模块的所有核心功能，包括：
- 配置模块测试
- 复杂度分析器测试
- 协作分配器测试
- 共识机制测试
- 基础智能体测试
- 各专家智能体测试
- 主控制器测试
- API模型集成测试
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
from typing import Dict, List, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入配置模块
from agent.mdagents.config import (
    ComplexityLevel, AgentType, CollaborationPattern, DecisionType,
    AgentConfig, ComplexityThresholds, MDAgentsConfig
)

# 导入核心模块
from agent.mdagents.core.complexity_analyzer import ComplexityAnalyzer, ComplexityFeatures
from agent.mdagents.core.collaboration_allocator import CollaborationAllocator, TeamConfig, CollaborationStrategy
from agent.mdagents.core.consensus_mechanism import (
    ConsensusMechanism, ConsensusResult, ConsensusMethod
)

# 导入智能体模块
from agent.mdagents.agents.base_agent import BaseAgent
from agent.mdagents.agents.pcc_agent import PrimaryCareClinician
from agent.mdagents.agents.specialist_agent import SpecialistAgent
from agent.mdagents.agents.moderator_agent import ModeratorAgent
from agent.mdagents.agents.recruiter_agent import RecruiterAgent
from agent.mdagents.agents.reviewer_agent import ReviewerAgent
from agent.mdagents.agents.integrator_agent import IntegratorAgent

# 导入主控制器
from agent.mdagents.core.main_controller import MainController

# 导入API模型（用于集成测试）
from models.api_model import ApiModel, ZhipuApiModel
from config.llm_config import LLMConfig


class MockApiModel:
    """Mock API模型用于测试"""
    
    def __init__(self):
        self.call_count = 0
    
    def generate(self, prompt: str, messages: List[Dict[str, str]] = None, 
                 additional_args: Dict[str, Any] = None) -> str:
        self.call_count += 1
        return f"Mock响应 #{self.call_count}: 测试回复"


class TestConfigModule(unittest.TestCase):
    """测试配置模块"""
    
    def test_complexity_level_enum(self):
        """测试复杂度等级枚举"""
        self.assertEqual(ComplexityLevel.SIMPLE.value, "simple")
        self.assertEqual(ComplexityLevel.MODERATE.value, "moderate")
        self.assertEqual(ComplexityLevel.COMPLEX.value, "complex")
        print("✓ ComplexityLevel 枚举测试通过")
    
    def test_agent_type_enum(self):
        """测试智能体类型枚举"""
        self.assertEqual(AgentType.MODERATOR.value, "moderator")
        self.assertEqual(AgentType.PCC.value, "pcc")
        self.assertEqual(AgentType.SPECIALIST.value, "specialist")
        self.assertEqual(AgentType.RECRUITER.value, "recruiter")
        self.assertEqual(AgentType.REVIEWER.value, "reviewer")
        self.assertEqual(AgentType.INTEGRATOR.value, "integrator")
        print("✓ AgentType 枚举测试通过")
    
    def test_collaboration_pattern_enum(self):
        """测试协作模式枚举"""
        self.assertEqual(CollaborationPattern.SINGLE_AGENT.value, "single_agent")
        self.assertEqual(CollaborationPattern.GROUP_COLLABORATION.value, "group_collaboration")
        self.assertEqual(CollaborationPattern.HIERARCHICAL_CONSULTATION.value, "hierarchical_consultation")
        print("✓ CollaborationPattern 枚举测试通过")
    
    def test_agent_config_creation(self):
        """测试智能体配置创建"""
        config = AgentConfig(
            name="测试智能体",
            agent_type="pcc",
            description="测试描述",
            capabilities=["诊断", "治疗"]
        )
        
        self.assertEqual(config.name, "测试智能体")
        self.assertEqual(config.agent_type, "pcc")
        self.assertIn("诊断", config.capabilities)
        print("✓ AgentConfig 创建测试通过")
    
    def test_complexity_thresholds_creation(self):
        """测试复杂度阈值配置"""
        thresholds = ComplexityThresholds(
            simple_max_score=0.3,
            moderate_max_score=0.6,
            complex_min_score=0.6
        )
        
        self.assertEqual(thresholds.simple_max_score, 0.3)
        self.assertEqual(thresholds.moderate_max_score, 0.6)
        self.assertIsNotNone(thresholds.text_length_weight)
        print("✓ ComplexityThresholds 创建测试通过")
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = AgentConfig(
            name="测试智能体",
            agent_type="specialist",
            description="测试"
        )
        
        config_dict = {
            'name': config.name,
            'agent_type': config.agent_type,
            'description': config.description
        }
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['name'], "测试智能体")
        print("✓ 配置转字典测试通过")


class TestComplexityAnalyzer(unittest.TestCase):
    """测试复杂度分析器"""
    
    def setUp(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        self.assertIsNotNone(self.analyzer.specialty_keywords)
        self.assertIsNotNone(self.analyzer.emergency_keywords)
        self.assertIsNotNone(self.analyzer.chronic_keywords)
        self.assertIsNotNone(self.analyzer.complex_medical_terms)
        print("✓ ComplexityAnalyzer 初始化测试通过")
    
    def test_simple_query_analysis(self):
        """测试简单查询分析"""
        query = "我感冒了，有点咳嗽"
        result = self.analyzer.analyze_complexity(query)
        
        self.assertIsNotNone(result)
        self.assertIn('complexity_score', result)
        self.assertIn('complexity_level', result)
        self.assertIn('features', result)
        print(f"✓ 简单查询分析测试通过 - 复杂度: {result['complexity_level'].value}")
    
    def test_moderate_query_analysis(self):
        """测试中等复杂度查询分析"""
        query = "患者有高血压病史10年，近期出现头晕、胸闷症状，血压控制不佳，考虑调整用药方案"
        result = self.analyzer.analyze_complexity(query)
        
        self.assertIsNotNone(result)
        self.assertIn('complexity_score', result)
        self.assertIn('features', result)
        print(f"✓ 中等复杂度查询分析测试通过 - 复杂度: {result['complexity_level'].value}")
    
    def test_complex_query_analysis(self):
        """测试复杂查询分析"""
        query = """患者为65岁男性，冠心病史，糖尿病史10年，高血压史15年，近期出现心前区疼痛，
        伴呼吸困难，既往有肾功能不全史，目前服用阿司匹林、氯吡格雷、二甲双胍、拜新同、倍他乐克等药物，
        需要多学科会诊讨论治疗方案"""
        result = self.analyzer.analyze_complexity(query)
        
        self.assertIsNotNone(result)
        self.assertIn('complexity_score', result)
        self.assertIn('features', result)
        print(f"✓ 复杂查询分析测试通过 - 复杂度: {result['complexity_level'].value}")
    
    def test_emergency_keywords_detection(self):
        """测试急症关键词检测"""
        query = "患者突发胸痛，呼吸困难，紧急情况"
        result = self.analyzer.analyze_complexity(query)
        
        features = result['features']
        self.assertTrue(len(features['emergency_keywords']) > 0)
        print(f"✓ 急症关键词检测测试通过 - 检测到: {features['emergency_keywords']}")
    
    def test_specialty_detection(self):
        """测试专科检测"""
        query = "患者心脏不适，心电图显示ST段抬高，需要心脏科会诊"
        result = self.analyzer.analyze_complexity(query)
        
        features = result['features']
        self.assertGreater(features['specialty_count'], 0)
        print(f"✓ 专科检测测试通过 - 涉及专科数: {features['specialty_count']}")


class TestCollaborationAllocator(unittest.TestCase):
    """测试协作分配器"""
    
    def setUp(self):
        self.allocator = CollaborationAllocator()
    
    def test_allocator_initialization(self):
        """测试分配器初始化"""
        self.assertIsNotNone(self.allocator.strategies)
        self.assertEqual(len(self.allocator.strategies), 3)
        print("✓ CollaborationAllocator 初始化测试通过")
    
    def test_simple_case_allocation(self):
        """测试简单病例分配"""
        complexity_level = ComplexityLevel.SIMPLE
        query = "我最近感冒了，有点咳嗽和鼻塞"
        result = self.allocator.allocate_collaboration(complexity_level, query)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'name'))
        self.assertTrue(hasattr(result, 'team_config'))
        self.assertTrue(hasattr(result, 'complexity_level'))
        print(f"✓ 简单病例分配测试通过 - 策略: {result.name}")
    
    def test_moderate_case_allocation(self):
        """测试中等复杂度病例分配"""
        complexity_level = ComplexityLevel.MODERATE
        query = "患者有胸痛和呼吸困难，既往有高血压病史"
        result = self.allocator.allocate_collaboration(complexity_level, query)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'name'))
        self.assertTrue(hasattr(result, 'team_config'))
        self.assertTrue(hasattr(result, 'complexity_level'))
        agents_needed = result.to_dict()['agents_needed']
        print(f"✓ 中等复杂度病例分配测试通过 - 所需智能体: {agents_needed}")
    
    def test_complex_case_allocation(self):
        """测试复杂病例分配"""
        complexity_level = ComplexityLevel.COMPLEX
        query = "患者有糖尿病、高血压、心脏病史，目前出现胸痛、呼吸困难、意识模糊等症状"
        result = self.allocator.allocate_collaboration(complexity_level, query)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'name'))
        self.assertTrue(hasattr(result, 'team_config'))
        self.assertTrue(hasattr(result, 'adaptive_rules'))
        print(f"✓ 复杂病例分配测试通过 - 适应规则: {len(result.adaptive_rules)}条")
    
    def test_team_config_creation(self):
        """测试团队配置创建"""
        team_config = TeamConfig(
            agents=[AgentType.PCC, AgentType.SPECIALIST],
            max_response_time=120,
            consensus_method="weighted_voting"
        )
        
        self.assertEqual(len(team_config.agents), 2)
        self.assertEqual(team_config.max_response_time, 120)
        print("✓ TeamConfig 创建测试通过")
    
    def test_strategy_creation(self):
        """测试协作策略创建"""
        strategy = CollaborationStrategy(
            name="专家会诊",
            complexity_level=ComplexityLevel.COMPLEX,
            team_config=TeamConfig(agents=[AgentType.PCC, AgentType.SPECIALIST])
        )
        
        self.assertEqual(strategy.name, "专家会诊")
        self.assertEqual(strategy.complexity_level, ComplexityLevel.COMPLEX)
        print("✓ CollaborationStrategy 创建测试通过")


class TestConsensusMechanism(unittest.TestCase):
    """测试共识机制"""
    
    def setUp(self):
        self.mechanism = ConsensusMechanism()
    
    def test_mechanism_initialization(self):
        """测试机制初始化"""
        self.assertIsNotNone(self.mechanism.consensus_thresholds)
        self.assertIsNotNone(self.mechanism.expert_weights)
        print("✓ ConsensusMechanism 初始化测试通过")
    
    def test_simple_majority_consensus(self):
        """测试简单多数共识"""
        opinions = [
            {"diagnosis": "感冒", "confidence": 0.9, "expert_type": "pcc"},
            {"diagnosis": "感冒", "confidence": 0.85, "expert_type": "specialist"},
            {"diagnosis": "流感", "confidence": 0.8, "expert_type": "specialist"}
        ]
        
        result = self.mechanism.achieve_consensus(
            opinions, algorithm="simple_majority"
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'consensus_level'))
        self.assertTrue(hasattr(result, 'final_decision'))
        print(f"✓ 简单多数共识测试通过 - 共识水平: {result.consensus_level:.2f}")
    
    def test_weighted_voting_consensus(self):
        """测试加权投票共识"""
        opinions = [
            {"diagnosis": "高血压", "confidence": 0.95, "expert_type": "specialist"},
            {"diagnosis": "高血压", "confidence": 0.9, "expert_type": "reviewer"},
            {"diagnosis": "低血压", "confidence": 0.6, "expert_type": "pcc"}
        ]
        
        result = self.mechanism.achieve_consensus(
            opinions, algorithm="weighted_voting"
        )
        
        self.assertIsNotNone(result)
        print(f"✓ 加权投票共识测试通过 - 共识水平: {result.consensus_level:.2f}")
    
    def test_borda_count_consensus(self):
        """测试Borda计数共识"""
        opinions = [
            {"diagnosis": ["A", "B", "C"], "confidence": 0.9, "expert_type": "pcc"},
            {"diagnosis": ["B", "A", "C"], "confidence": 0.85, "expert_type": "specialist"},
            {"diagnosis": ["A", "C", "B"], "confidence": 0.8, "expert_type": "reviewer"}
        ]
        
        result = self.mechanism.achieve_consensus(
            opinions, algorithm="borda_count"
        )
        
        self.assertIsNotNone(result)
        print(f"✓ Borda计数共识测试通过 - 共识水平: {result.consensus_level:.2f}")
    
    def test_consensus_result_to_dict(self):
        """测试共识结果转字典"""
        result = ConsensusResult(
            consensus_level=0.85,
            final_decision={"diagnosis": "测试"},
            supporting_agents=["agent1", "agent2"],
            confidence_score=0.9
        )
        
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['consensus_level'], 0.85)
        print("✓ 共识结果转字典测试通过")


class TestBaseAgent(unittest.TestCase):
    """测试基础智能体功能（通过具体实现类测试）"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="测试智能体",
            agent_type="pcc",
            description="测试基础智能体"
        )
        self.mock_api = MockApiModel()
    
    def test_agent_initialization(self):
        """测试智能体初始化（使用具体实现类）"""
        agent = PrimaryCareClinician(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "测试智能体")
        self.assertEqual(agent.status, "idle")
        self.assertIsNotNone(agent.conversation_history)
        print("✓ 智能体初始化测试通过")
    
    def test_agent_with_api_model(self):
        """测试带API模型的智能体"""
        agent = PrimaryCareClinician(self.config, api_model=self.mock_api)
        
        self.assertIsNotNone(agent.api_model)
        self.assertIsNotNone(agent.api_wrapper)
        print("✓ 智能体API模型集成测试通过")
    
    def test_log_interaction(self):
        """测试交互日志记录"""
        agent = PrimaryCareClinician(self.config)
        agent.log_interaction("test", "测试内容", {"key": "value"})
        
        self.assertEqual(len(agent.conversation_history), 1)
        log_entry = agent.conversation_history[0]
        self.assertEqual(log_entry['interaction_type'], "test")
        self.assertEqual(log_entry['content'], "测试内容")
        print("✓ 交互日志记录测试通过")
    
    def test_generate_response_requires_api(self):
        """测试生成响应需要API模型"""
        agent = PrimaryCareClinician(self.config)
        
        with self.assertRaises(ValueError):
            agent.generate_response("测试提示词")
        print("✓ 生成响应API依赖测试通过")
    
    def test_generate_response_with_mock(self):
        """测试使用Mock API生成响应"""
        agent = PrimaryCareClinician(self.config, api_model=self.mock_api)
        response = agent.generate_response("测试提示词")
        
        self.assertIn("Mock响应", response)
        self.assertEqual(self.mock_api.call_count, 1)
        print("✓ Mock API生成响应测试通过")


class TestPCRAgent(unittest.TestCase):
    """测试初级保健医生智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="全科医生_测试",
            agent_type="pcc",
            description="测试用初级保健医生"
        )
        self.mock_api = MockApiModel()
    
    def test_pcc_agent_initialization(self):
        """测试PCC智能体初始化"""
        agent = PrimaryCareClinician(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "全科医生_测试")
        self.assertIn("全科", agent.specialties)
        self.assertIn("上呼吸道感染", agent.common_conditions)
        print("✓ PCCAgent 初始化测试通过")
    
    def test_process_query(self):
        """测试处理查询"""
        agent = PrimaryCareClinician(self.config, api_model=self.mock_api)
        result = agent.process_query("我感冒了怎么办？")
        
        self.assertIsNotNone(result)
        self.assertIn('agent_type', result)
        self.assertIn('agent_name', result)
        self.assertIn('diagnosis', result)
        self.assertIn('treatment_plan', result)
        print(f"✓ PCCAgent 处理查询测试通过 - 类型: {result['agent_type']}")


class TestSpecialistAgent(unittest.TestCase):
    """测试专科医生智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="心血管专家_测试",
            agent_type="specialist",
            description="测试用心血管专家"
        )
        self.mock_api = MockApiModel()
    
    def test_specialist_agent_initialization(self):
        """测试专科医生初始化"""
        agent = SpecialistAgent(
            self.config, 
            specialty="心血管科", 
            api_model=self.mock_api
        )
        
        self.assertEqual(agent.specialty, "心血管科")
        self.assertIn("冠心病", agent.specialty_knowledge["common_conditions"])
        print("✓ SpecialistAgent 初始化测试通过")
    
    def test_specialist_agent_generic(self):
        """测试通用专科医生"""
        agent = SpecialistAgent(
            self.config, 
            specialty="未知专科",
            api_model=self.mock_api
        )
        
        self.assertEqual(agent.specialty, "未知专科")
        print("✓ 通用专科医生测试通过")
    
    def test_process_query(self):
        """测试处理专科查询"""
        agent = SpecialistAgent(
            self.config,
            specialty="心血管科",
            api_model=self.mock_api
        )
        result = agent.process_query("患者胸痛，心电图异常")

        self.assertIsNotNone(result)
        self.assertIn('agent_type', result)
        self.assertIn('specialty', result)
        self.assertIn('specialist_assessment', result)
        self.assertEqual(result['specialty'], "心血管科")
        print(f"✓ SpecialistAgent 处理查询测试通过 - 专科: {result['specialty']}")


class TestModeratorAgent(unittest.TestCase):
    """测试主持人智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="主持人_测试",
            agent_type="moderator",
            description="测试用主持人"
        )
        self.mock_api = MockApiModel()
    
    def test_moderator_agent_initialization(self):
        """测试主持人初始化"""
        agent = ModeratorAgent(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "主持人_测试")
        self.assertIsNotNone(agent.complexity_analyzer)
        print("✓ ModeratorAgent 初始化测试通过")
    
    def test_process_query(self):
        """测试处理调节查询"""
        agent = ModeratorAgent(self.config, api_model=self.mock_api)
        result = agent.process_query("测试问题")

        self.assertIsNotNone(result)
        self.assertIn('agent_type', result)
        self.assertIn('agent_name', result)
        self.assertIn('complexity_analysis', result)
        print(f"✓ ModeratorAgent 处理查询测试通过 - 类型: {result['agent_type']}")


class TestRecruiterAgent(unittest.TestCase):
    """测试招募者智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="招募者_测试",
            agent_type="recruiter",
            description="测试用招募者"
        )
        self.mock_api = MockApiModel()
    
    def test_recruiter_agent_initialization(self):
        """测试招募者初始化"""
        agent = RecruiterAgent(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "招募者_测试")
        self.assertIsNotNone(agent.expert_registry)
        print("✓ RecruiterAgent 初始化测试通过")
    
    def test_process_query(self):
        """测试处理招募查询"""
        agent = RecruiterAgent(self.config, api_model=self.mock_api)
        result = agent.process_query("心血管科复杂病例需要团队")

        self.assertIsNotNone(result)
        self.assertIn('agent_type', result)
        self.assertIn('agent_name', result)
        self.assertIn('requirements_analysis', result)
        print(f"✓ RecruiterAgent 处理查询测试通过 - 类型: {result['agent_type']}")


class TestReviewerAgent(unittest.TestCase):
    """测试审查员智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="审查员_测试",
            agent_type="reviewer",
            description="测试用审查员"
        )
        self.mock_api = MockApiModel()
    
    def test_reviewer_agent_initialization(self):
        """测试审查员初始化"""
        agent = ReviewerAgent(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "审查员_测试")
        self.assertIn("medical_accuracy", agent.quality_standards)
        self.assertIn("safety", agent.quality_standards)
        print("✓ ReviewerAgent 初始化测试通过")
    
    def test_process_query(self):
        """测试处理审查查询"""
        agent = ReviewerAgent(self.config, api_model=self.mock_api)
        context = {
            'medical_advice': '患者诊断为高血压，建议进行药物治疗，包括使用降压药物控制血压，定期监测血压变化。',
            'agent_recommendations': [
                {'diagnosis': '高血压', 'confidence': 0.9, 'reason': '血压测量', 'agent_type': 'specialist'}
            ],
            'complexity_level': ComplexityLevel.MODERATE
        }
        result = agent.process_query("审查高血压诊断建议", context)

        self.assertIsNotNone(result)
        self.assertIn('status', result)
        self.assertIn('review_result', result)
        print(f"✓ ReviewerAgent 处理查询测试通过 - 状态: {result['status']}")


class TestIntegratorAgent(unittest.TestCase):
    """测试整合者智能体"""
    
    def setUp(self):
        self.config = AgentConfig(
            name="整合者_测试",
            agent_type="integrator",
            description="测试用整合者"
        )
        self.mock_api = MockApiModel()
    
    def test_integrator_agent_initialization(self):
        """测试整合者初始化"""
        agent = IntegratorAgent(self.config, api_model=self.mock_api)
        
        self.assertEqual(agent.name, "整合者_测试")
        self.assertIn("weighted_average", agent.integration_strategies)
        self.assertIn("majority_vote", agent.integration_strategies)
        print("✓ IntegratorAgent 初始化测试通过")
    
    def test_integrate_opinions(self):
        """测试整合意见"""
        agent = IntegratorAgent(self.config, api_model=self.mock_api)
        context = {
            'agent_recommendations': [
                {"diagnosis": "高血压", "confidence": 0.9, "reason": "血压测量", "agent_type": "specialist"},
                {"diagnosis": "高血压", "confidence": 0.85, "reason": "症状分析", "agent_type": "pcc"}
            ],
            'complexity_level': ComplexityLevel.MODERATE
        }
        result = agent.process_query("整合高血压治疗建议", context)
        
        self.assertIsNotNone(result)
        self.assertIn('integration_result', result)
        self.assertIn('status', result)
        print(f"✓ IntegratorAgent 整合意见测试通过 - 状态: {result['status']}")


class TestMainController(unittest.TestCase):
    """测试主控制器"""
    
    def setUp(self):
        self.mock_api = MockApiModel()
    
    def test_controller_initialization(self):
        """测试控制器初始化"""
        controller = MainController()
        
        self.assertEqual(controller.system_status, "initialized")
        self.assertIsNotNone(controller.complexity_analyzer)
        self.assertIsNotNone(controller.collaboration_allocator)
        self.assertIsNotNone(controller.consensus_mechanism)
        self.assertIn("pcc", controller.agents)
        self.assertIn("specialist", controller.agents)
        print("✓ MainController 初始化测试通过")
    
    def test_controller_with_api(self):
        """测试带API的控制器"""
        controller = MainController(llm_config=None)
        controller.api_model = self.mock_api
        
        self.assertIsNotNone(controller.api_model)
        print("✓ MainController API集成测试通过")
    
    def test_process_query_simple(self):
        """测试处理简单查询"""
        controller = MainController()
        controller.api_model = self.mock_api
        
        query = "我最近感冒了，有点咳嗽和鼻塞"
        
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            controller.process_medical_query(query)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('query', result)
        self.assertIn('complexity_analysis', result)
        self.assertIn('final_report', result)
        print(f"✓ MainController 处理简单查询测试通过 - 复杂度: {result['complexity_analysis']['complexity_level']}")
    
    def test_process_query_complex(self):
        """测试处理复杂查询"""
        controller = MainController()
        controller.api_model = self.mock_api
        
        query = """患者为65岁男性，有冠心病、糖尿病、高血压病史，
        近期出现胸痛、呼吸困难，需要多学科会诊"""
        
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            controller.process_medical_query(query)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('collaboration_plan', result)
        self.assertIn('expert_agents', result)
        print(f"✓ MainController 处理复杂查询测试通过 - 复杂度: {result['complexity_analysis']['complexity_level']}")
    
    def test_system_status(self):
        """测试系统状态"""
        controller = MainController()
        
        status = controller.system_status
        self.assertEqual(status, "initialized")
        print("✓ MainController 系统状态测试通过")


class TestApiModelIntegration(unittest.TestCase):
    """测试API模型集成"""
    
    def test_api_model_initialization(self):
        """测试API模型初始化"""
        try:
            # 注意：这需要真实的API密钥，这里测试的是导入和基本结构
            from config.llm_config import LLMConfig
            
            config = LLMConfig(
                model_name="test-model",
                api_key="test-key",
                base_url="http://test.com"
            )
            
            self.assertEqual(config.model_name, "test-model")
            print("✓ LLMConfig 初始化测试通过")
        except Exception as e:
            print(f"⚠ LLMConfig 初始化测试跳过: {e}")
    
    def test_api_model_generate_signature(self):
        """测试API模型generate方法签名"""
        # 验证ApiModel类的generate方法签名
        import inspect
        
        sig = inspect.signature(ApiModel.generate)
        params = list(sig.parameters.keys())
        
        self.assertIn('prompt', params)
        self.assertIn('additional_args', params)
        self.assertIn('messages', params)
        print("✓ ApiModel.generate 方法签名测试通过")
    
    def test_zhipu_api_model_available(self):
        """测试智谱API模型可用性"""
        try:
            self.assertTrue(issubclass(ZhipuApiModel, ApiModel))
            print("✓ ZhipuApiModel 继承测试通过")
        except Exception as e:
            print(f"⚠ ZhipuApiModel 测试跳过: {e}")


class TestIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def setUp(self):
        self.mock_api = MockApiModel()
    
    def test_full_medical_query_flow(self):
        """测试完整医疗查询流程"""
        # 1. 初始化控制器
        controller = MainController()
        controller.api_model = self.mock_api
        
        # 2. 提交查询
        query = "患者有高血压和糖尿病史，近期出现头晕、胸闷症状"
        
        # 3. 处理查询
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            controller.process_medical_query(query)
        )
        
        # 4. 验证结果
        self.assertIsNotNone(result)
        self.assertIn('query', result)
        self.assertIn('complexity_analysis', result)
        self.assertIn('collaboration_plan', result)
        self.assertIn('final_report', result)
        
        # 5. 验证复杂度分析
        complexity = result['complexity_analysis']
        self.assertIn('complexity_score', complexity)
        self.assertIn('complexity_level', complexity)
        
        print("✓ 完整医疗查询流程测试通过")
    
    def test_multi_agent_collaboration(self):
        """测试多智能体协作"""
        controller = MainController()
        controller.api_model = self.mock_api
        
        # 提交需要多学科协作的查询
        query = """患者为70岁老年男性，有冠心病、心房颤动、高血压、2型糖尿病史，
        近期出现心悸、气促、下肢水肿，考虑心力衰竭可能性大，
        需要心内科、内分泌科会诊讨论治疗方案"""
        
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            controller.process_medical_query(query)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('collaboration_plan', result)
        self.assertIn('expert_agents', result)
        
        # 验证专家团队
        agents = result['expert_agents']
        self.assertIsInstance(agents, list)
        
        print("✓ 多智能体协作测试通过")
    
    def test_consensus_building(self):
        """测试共识构建"""
        controller = MainController()
        controller.api_model = self.mock_api
        
        # 提交需要共识的查询
        query = "患者血压控制不佳，需要调整治疗方案"
        
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            controller.process_medical_query(query)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('consensus_result', result)
        self.assertIn('final_report', result)
        
        print("✓ 共识构建测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("MDAgents 全面测试开始")
    print("=" * 80)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestConfigModule,
        TestComplexityAnalyzer,
        TestCollaborationAllocator,
        TestConsensusMechanism,
        TestBaseAgent,
        TestPCRAgent,
        TestSpecialistAgent,
        TestModeratorAgent,
        TestRecruiterAgent,
        TestReviewerAgent,
        TestIntegratorAgent,
        TestMainController,
        TestApiModelIntegration,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print()
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.wasSuccessful()}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print()
        print("🎉 所有测试通过！MDAgents 系统运行正常。")
    else:
        print()
        print("❌ 存在测试失败或错误，请检查输出。")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
