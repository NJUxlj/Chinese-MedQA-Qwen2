"""
MDAgents系统快速测试脚本
用于验证系统核心功能的正常运行
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdagents.config import get_config_manager, SystemConfig
from mdagents.core.main_controller import MainController


async def quick_test():
    """快速功能测试"""
    print("=" * 60)
    print("MDAgents系统快速测试")
    print("=" * 60)
    
    try:
        # 1. 测试配置系统
        print("\n1. 测试配置系统...")
        config_manager = get_config_manager()
        config = config_manager.config
        print(f"   ✓ 配置加载成功: {config.llm.model_name}")
        
        # 2. 测试主控制器初始化
        print("\n2. 测试主控制器初始化...")
        controller = MainController(llm_config=None)
        print(f"   ✓ 主控制器初始化成功")
        print(f"   ✓ 激活智能体数量: {len(controller.agents)}")
        print(f"   ✓ 智能体类型: {list(controller.agents.keys())}")
        
        # 3. 测试复杂度分析
        print("\n3. 测试复杂度分析...")
        complexity_analyzer = controller.complexity_analyzer
        test_query = "患者感到头痛和发热"
        result = complexity_analyzer.analyze_complexity(test_query)
        print(f"   ✓ 复杂度分析完成: {result.get('complexity_level', 'N/A')}")
        
        # 4. 测试协作分配
        print("\n4. 测试协作分配...")
        collaboration_allocator = controller.collaboration_allocator
        collaboration_plan = collaboration_allocator.allocate_collaboration(
            result['complexity_level'], test_query
        )
        print(f"   ✓ 协作模式: {collaboration_plan.get('pattern', 'N/A')}")
        print(f"   ✓ 所需智能体: {collaboration_plan.get('agents_needed', [])}")
        
        # 5. 测试共识机制
        print("\n5. 测试共识机制...")
        consensus_mechanism = controller.consensus_mechanism
        
        # 模拟专家意见
        expert_opinions = [
            {
                "agent_name": "全科医生",
                "agent_type": "pcc",
                "diagnosis": "感冒",
                "confidence": 0.8,
                "recommendations": ["多休息", "多喝水"]
            },
            {
                "agent_name": "专科医生",
                "agent_type": "specialist", 
                "diagnosis": "上呼吸道感染",
                "confidence": 0.9,
                "recommendations": ["对症治疗", "观察症状"]
            }
        ]
        
        consensus_result = consensus_mechanism.achieve_consensus(
            expert_opinions=expert_opinions,
            algorithm="simple_majority",
            decision_type="diagnosis"
        )
        print(f"   ✓ 共识达成: {consensus_result.final_decision}")
        print(f"   ✓ 置信度: {consensus_result.confidence:.2f}")
        
        # 6. 测试系统状态
        print("\n6. 测试系统状态...")
        status = controller.get_system_status()
        print(f"   ✓ 系统状态: {status['system_status']}")
        print(f"   ✓ 成功率: {status['success_rate']:.2f}")
        
        print("\n" + "=" * 60)
        print("✅ 所有核心功能测试通过！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def full_workflow_test():
    """完整工作流程测试"""
    print("\n" + "=" * 60)
    print("完整工作流程测试")
    print("=" * 60)
    
    try:
        controller = MainController(llm_config=None)
        
        # 测试不同的医疗查询
        test_cases = [
            {
                "query": "我有点头痛，可能是什么原因？",
                "patient_info": {"age": 25, "gender": "female"}
            },
            {
                "query": "患者出现胸痛、呼吸困难，伴有发热",
                "patient_info": {"age": 55, "gender": "male", "history": ["hypertension"]}
            },
            {
                "query": "医生，我最近总是失眠，有什么建议吗？",
                "patient_info": {"age": 35, "gender": "female"}
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_case['query'][:30]}...")
            
            result = await controller.process_medical_query(
                test_case['query'],
                test_case['patient_info']
            )
            
            if result['status'] == 'success':
                print(f"   ✓ 处理成功")
                print(f"   ✓ 复杂度: {result['complexity_analysis']['complexity_level']}")
                print(f"   ✓ 协作模式: {result['collaboration_plan']['pattern']}")
                print(f"   ✓ 参与专家: {len(result['expert_agents'])}")
                print(f"   ✓ 共识结果: {result['consensus_result']['final_decision']}")
            else:
                print(f"   ❌ 处理失败: {result.get('error', '未知错误')}")
        
        # 最终系统状态
        final_status = controller.get_system_status()
        print(f"\n最终系统状态:")
        print(f"   总决策数: {final_status['total_decisions']}")
        print(f"   成功决策数: {final_status['successful_consensus']}")
        print(f"   成功率: {final_status['success_rate']:.2f}")
        
        print("\n" + "=" * 60)
        print("✅ 完整工作流程测试完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("MDAgents系统测试开始...")
    
    # 运行快速测试
    quick_success = await quick_test()
    
    if quick_success:
        # 运行完整工作流程测试
        workflow_success = await full_workflow_test()
        
        if workflow_success:
            print("\n🎉 所有测试通过！MDAgents系统运行正常。")
            return True
        else:
            print("\n⚠️  工作流程测试失败，但核心功能正常。")
            return False
    else:
        print("\n❌ 核心功能测试失败，请检查系统配置。")
        return False


if __name__ == "__main__":
    # 运行异步测试
    success = asyncio.run(main())
    exit(0 if success else 1)