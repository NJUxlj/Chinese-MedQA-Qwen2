#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDAgents使用示例
展示如何使用医疗决策制定多智能体系统
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.mdagents.core.main_controller import MainController
from agent.mdagents.core.complexity_analyzer import ComplexityAnalyzer
from agent.mdagents.core.collaboration_allocator import CollaborationAllocator
from agent.mdagents.core.consensus_mechanism import ConsensusMechanism, ConsensusMethod
from agent.mdagents.config.system_config import (
    SystemConfig, LLMConfig, ConsensusConfig, AgentPoolConfig, 
    ComplexityConfig, LoggingConfig
)


class MDAgentsExamples:
    """MDAgents使用示例类"""

    def __init__(self):
        self.controller = None

    async def setup_system(self, llm_config: Optional[Dict[str, Any]] = None) -> MainController:
        """初始化系统"""
        print("🔧 初始化MDAgents系统...")
        
        # 基础LLM配置
        if llm_config is None:
            llm_config = {
                "model_name": "medical-llm",
                "api_base_url": "http://localhost:8000",
                "api_key": "demo_key",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 30,
                "retry_times": 3
            }
        
        # 创建系统配置
        config = SystemConfig(
            llm=LLMConfig.from_dict(llm_config),
            agent_pool=AgentPoolConfig(
                pcc_agents=1,
                specialist_agents=3,
                moderator_agents=1,
                recruiter_agents=1,
                reviewer_agents=1,
                integrator_agents=1
            ),
            consensus=ConsensusConfig(
                default_algorithm="weighted_voting",
                min_expert_count=2,
                consensus_thresholds={
                    "unanimous": 1.0,
                    "super_majority": 0.8,
                    "majority": 0.6,
                    "plurality": 0.4,
                    "minimum": 0.3
                }
            ),
            complexity=ComplexityConfig(
                enable_ai_analysis=True,
                manual_complexity_mapping={
                    "急诊": "COMPLEX",
                    "复杂": "COMPLEX",
                    "罕见": "COMPLEX",
                    "严重": "COMPLEX",
                    "普通": "SIMPLE",
                    "常见": "MODERATE",
                    "一般": "SIMPLE"
                },
                complexity_keywords={
                    "SIMPLE": ["头痛", "感冒", "发烧", "咳嗽", "胃痛", "普通", "常见"],
                    "MODERATE": ["胸痛", "呼吸困难", "腹痛", "头晕", "失眠", "高血压"],
                    "COMPLEX": ["多系统疾病", "罕见病", "并发症", "急诊", "手术", "肿瘤"]
                }
            ),
            logging=LoggingConfig(
                level="INFO",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                enable_console=True
            )
        )
        
        # 初始化控制器
        self.controller = MainController(llm_config=config.llm)
        
        print("✅ 系统初始化完成")
        return self.controller

    async def example_basic_consultation(self):
        """示例1: 基础医疗咨询"""
        print("\n" + "=" * 60)
        print("示例1: 基础医疗咨询")
        print("=" * 60)
        
        controller = await self.setup_system()
        
        # 简单的症状咨询
        query = "医生，我最近总是头痛，特别是下午的时候，可能是什么原因？"
        patient_info = {
            "age": 28,
            "gender": "female",
            "history": ["无特殊病史"],
            "symptoms": ["头痛", "疲劳"]
        }
        
        print(f"📝 查询: {query}")
        print(f"👤 患者信息: {patient_info}")
        
        result = await controller.process_medical_query(query, patient_info)
        
        if result["status"] == "success":
            print("✅ 处理成功!")
            print(f"📊 复杂度等级: {result['complexity_analysis']['complexity_level']}")
            print(f"🤝 协作模式: {result['collaboration_plan']['pattern']}")
            print(f"👨‍⚕️ 参与专家: {len(result['expert_agents'])}")
            print(f"🎯 最终决策: {result['consensus_result']['final_decision']}")
        else:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")

    async def example_complex_case(self):
        """示例2: 复杂病例处理"""
        print("\n" + "=" * 60)
        print("示例2: 复杂病例处理")
        print("=" * 60)
        
        controller = await self.setup_system()
        
        # 复杂的多症状病例
        query = """患者，男性，55岁，主诉胸痛3天，伴有呼吸困难、出汗，
                  既往有高血压病史10年，吸烟史30年。请问这可能是什么疾病？"""
        
        patient_info = {
            "age": 55,
            "gender": "male",
            "history": ["高血压10年", "吸烟30年"],
            "symptoms": ["胸痛", "呼吸困难", "出汗", "焦虑"],
            "vital_signs": {
                "blood_pressure": "160/100",
                "heart_rate": "95",
                "temperature": "37.2"
            }
        }
        
        print(f"📝 查询: {query}")
        print(f"👤 患者信息: {patient_info}")
        
        result = await controller.process_medical_query(query, patient_info)
        
        if result["status"] == "success":
            print("✅ 处理成功!")
            print(f"📊 复杂度等级: {result['complexity_analysis']['complexity_level']}")
            print(f"📋 复杂度分析: {result['complexity_analysis']['explanation']}")
            print(f"🤝 协作模式: {result['collaboration_plan']['pattern']}")
            print(f"👨‍⚕️ 参与专家: {', '.join(result['expert_agents'])}")
            print(f"🎯 最终决策: {result['consensus_result']['final_decision']}")
            print(f"💡 建议: {result['final_report']['recommendations']}")
        else:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")

    async def example_different_consensus_methods(self):
        """示例3: 不同共识机制的使用"""
        print("\n" + "=" * 60)
        print("示例3: 不同共识机制的使用")
        print("=" * 60)
        
        # 测试不同的共识机制
        consensus_methods = [
            (ConsensusMethod.SIMPLE_MAJORITY, "简单多数投票"),
            (ConsensusMethod.WEIGHTED_VOTING, "加权投票"),
            (ConsensusMethod.BORDA_COUNT, "Borda计数"),
            (ConsensusMethod.FUZZY_CONSENSUS, "模糊共识")
        ]
        
        query = "患者出现持续性腹痛，需要进行哪些检查？"
        patient_info = {"age": 45, "gender": "female"}
        
        for method, name in consensus_methods:
            print(f"\n🔄 测试共识机制: {name}")
            
            controller = await self.setup_system()
            
            # 临时修改默认共识方法
            controller.consensus_mechanism.default_method = method
            
            result = await controller.process_medical_query(query, patient_info)
            
            if result["status"] == "success":
                print(f"✅ {name} - 成功")
                print(f"🎯 决策: {result['consensus_result']['final_decision']}")
                print(f"📈 置信度: {result['consensus_result'].get('confidence_score', 'N/A')}")
            else:
                print(f"❌ {name} - 失败: {result.get('error', '未知错误')}")

    async def example_batch_processing(self):
        """示例4: 批量处理多个查询"""
        print("\n" + "=" * 60)
        print("示例4: 批量处理多个查询")
        print("=" * 60)
        
        controller = await self.setup_system()
        
        # 批量查询列表
        batch_queries = [
            {
                "query": "孩子发烧38.5度，需要去医院吗？",
                "patient_info": {"age": 8, "gender": "male", "symptoms": ["发热", "咳嗽"]}
            },
            {
                "query": "老年人关节疼痛，如何缓解？",
                "patient_info": {"age": 70, "gender": "female", "history": ["关节炎"]}
            },
            {
                "query": "女性月经不调，应该检查什么？",
                "patient_info": {"age": 25, "gender": "female", "symptoms": ["月经不调"]}
            }
        ]
        
        print(f"📋 处理 {len(batch_queries)} 个查询...")
        
        results = []
        for i, case in enumerate(batch_queries, 1):
            print(f"\n🔄 处理查询 {i}: {case['query'][:20]}...")
            
            result = await controller.process_medical_query(
                case["query"], 
                case["patient_info"]
            )
            
            if result["status"] == "success":
                print(f"✅ 查询 {i} - 成功")
                results.append({
                    "query": case["query"],
                    "complexity": result["complexity_analysis"]["complexity_level"],
                    "decision": result["consensus_result"]["final_decision"]
                })
            else:
                print(f"❌ 查询 {i} - 失败")
        
        # 显示批量处理结果摘要
        print("\n📊 批量处理结果摘要:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['complexity']} - {result['decision']}")

    async def example_system_monitoring(self):
        """示例5: 系统状态监控"""
        print("\n" + "=" * 60)
        print("示例5: 系统状态监控")
        print("=" * 60)
        
        controller = await self.setup_system()
        
        # 执行一些查询来积累统计信息
        test_queries = [
            ("轻微头痛", {"age": 30, "gender": "female"}),
            ("胸痛气短", {"age": 50, "gender": "male"})
        ]
        
        print("🔄 执行测试查询以积累统计信息...")
        
        for query, patient_info in test_queries:
            await controller.process_medical_query(query, patient_info)
        
        # 获取系统状态
        status = controller.get_system_status()
        
        print("📊 系统状态:")
        print(f"状态: {status['system_status']}")
        print(f"总决策数: {status['total_decisions']}")
        print(f"成功共识数: {status['successful_consensus']}")
        print(f"成功率: {status.get('success_rate', 0):.2%}")
        
        if status.get('active_collaboration'):
            print(f"当前活跃协作: {status['active_collaboration']['pattern']}")
        
        if status.get('recent_sessions'):
            print(f"最近会话数: {len(status['recent_sessions'])}")

    async def example_custom_configuration(self):
        """示例6: 自定义配置使用"""
        print("\n" + "=" * 60)
        print("示例6: 自定义配置使用")
        print("=" * 60)
        
        # 创建自定义配置
        custom_llm_config = {
            "model_name": "gpt-4",
            "api_base_url": "https://api.openai.com/v1",
            "api_key": "your_api_key_here",
            "max_tokens": 4000,
            "temperature": 0.7,
            "timeout": 60,
            "retry_times": 3
        }
        
        print("🔧 使用自定义配置初始化系统...")
        
        config = SystemConfig(
            llm=LLMConfig.from_dict(custom_llm_config),
            agent_pool=AgentPoolConfig(
                pcc_agents=1,
                specialist_agents=2,
                moderator_agents=1,
                recruiter_agents=1,
                reviewer_agents=1,
                integrator_agents=1
            ),
            consensus=ConsensusConfig(
                default_algorithm="weighted_voting",
                min_expert_count=2,
                consensus_thresholds={
                    "unanimous": 1.0,
                    "super_majority": 0.8,
                    "majority": 0.6,
                    "plurality": 0.4,
                    "minimum": 0.3
                }
            ),
            complexity=ComplexityConfig(
                enable_ai_analysis=True,
                manual_complexity_mapping={
                    "急诊": "COMPLEX",
                    "复杂": "COMPLEX",
                    "罕见": "COMPLEX",
                    "严重": "COMPLEX",
                    "普通": "SIMPLE",
                    "常见": "MODERATE",
                    "一般": "SIMPLE"
                },
                complexity_keywords={
                    "SIMPLE": ["头痛", "感冒", "发烧", "咳嗽", "胃痛", "普通", "常见"],
                    "MODERATE": ["胸痛", "呼吸困难", "腹痛", "头晕", "失眠", "高血压"],
                    "COMPLEX": ["多系统疾病", "罕见病", "并发症", "急诊", "手术", "肿瘤"]
                }
            ),
            logging=LoggingConfig(
                level="INFO",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                enable_console=True
            )
        )
        
        controller = MainController(llm_config=config.llm)
        
        # 使用自定义配置处理查询
        query = "如何使用自定义配置进行医疗咨询？"
        patient_info = {"age": 35, "gender": "female"}
        
        print(f"📝 查询: {query}")
        
        result = await controller.process_medical_query(query, patient_info)
        
        if result["status"] == "success":
            print("✅ 自定义配置处理成功!")
            print(f"🤝 使用的协作模式: {result['collaboration_plan']['pattern']}")
            print(f"🎯 最终决策: {result['consensus_result']['final_decision']}")
        else:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")


async def run_all_examples():
    """运行所有示例"""
    print("🏥 MDAgents医疗决策制定多智能体系统 - 使用示例")
    print("=" * 80)
    
    examples = MDAgentsExamples()
    
    try:
        # 运行各种示例
        await examples.example_basic_consultation()
        await examples.example_complex_case()
        await examples.example_different_consensus_methods()
        await examples.example_batch_processing()
        await examples.example_system_monitoring()
        await examples.example_custom_configuration()
        
        print("\n" + "=" * 80)
        print("🎉 所有示例运行完成!")
        print("📚 更多信息请参考:")
        print("   - API文档: /src/agent/mdagents/")
        print("   - 测试用例: /tests/mdagents/")
        print("   - 配置文件: /src/agent/mdagents/config/")
        
    except Exception as e:
        print(f"❌ 运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()


async def run_single_example():
    """运行单个示例（快速测试）"""
    print("🚀 MDAgents快速测试")
    print("=" * 50)
    
    examples = MDAgentsExamples()
    await examples.setup_system()
    await examples.example_basic_consultation()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MDAgents使用示例")
    parser.add_argument(
        "--mode", 
        choices=["all", "single"], 
        default="single",
        help="运行模式: all(所有示例) 或 single(单个示例)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "all":
        asyncio.run(run_all_examples())
    else:
        asyncio.run(run_single_example())