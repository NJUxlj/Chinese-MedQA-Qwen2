#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDAgents快速入门示例
演示最基本的使用方法
"""

import asyncio
import os
import sys

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.mdagents.core.main_controller import MainController


async def quick_start_example():
    """快速入门示例"""
    print("🏥 欢迎使用MDAgents医疗决策制定系统!")
    print("=" * 50)
    
    # 步骤1: 初始化系统
    print("1. 初始化系统...")
    controller = MainController(llm_config=None)  # 使用默认配置
    
    # 步骤2: 准备医疗查询
    query = "医生，我最近总是感到疲劳，可能是什么原因？"
    patient_info = {
        "age": 30,
        "gender": "female",
        "symptoms": ["疲劳", "头痛"],
        "history": ["无特殊病史"]
    }
    
    print(f"2. 查询: {query}")
    print(f"3. 患者信息: {patient_info}")
    
    # 步骤3: 处理查询
    print("4. 处理查询...")
    result = await controller.process_medical_query(query, patient_info)
    
    # 步骤4: 显示结果
    if result["status"] == "success":
        print("✅ 处理成功!")
        print("\n📊 结果摘要:")
        print(f"   • 复杂度等级: {result['complexity_analysis']['complexity_level']}")
        print(f"   • 协作模式: {result['collaboration_plan']['pattern']}")
        print(f"   • 参与专家: {len(result['expert_agents'])} 个")
        print(f"   • 最终决策: {result['consensus_result']['final_decision']}")
        print(f"   • 置信度: {result['consensus_result'].get('confidence_score', 'N/A')}")
        
        print("\n💡 建议:")
        recommendations = result['final_report']['recommendations']
        for i, rec in enumerate(recommendations[:3], 1):  # 显示前3条建议
            print(f"   {i}. {rec}")
    else:
        print(f"❌ 处理失败: {result.get('error', '未知错误')}")
    
    # 步骤5: 显示系统状态
    print("\n📈 系统状态:")
    status = controller.get_system_status()
    print(f"   • 总处理数: {status['total_decisions']}")
    print(f"   • 成功率: {status.get('success_rate', 0):.1%}")


async def multiple_queries_example():
    """多个查询示例"""
    print("\n" + "=" * 50)
    print("多查询处理示例")
    print("=" * 50)
    
    controller = MainController(llm_config=None)
    
    # 测试用例
    test_cases = [
        {
            "query": "我有点头晕，是怎么回事？",
            "patient": {"age": 25, "gender": "female"},
            "description": "轻微症状"
        },
        {
            "query": "患者胸痛、呼吸困难，需要紧急处理吗？",
            "patient": {"age": 55, "gender": "male", "history": ["高血压"]},
            "description": "复杂症状"
        },
        {
            "query": "失眠怎么办？",
            "patient": {"age": 35, "gender": "female"},
            "description": "常见问题"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔄 测试用例 {i}: {case['description']}")
        print(f"   查询: {case['query'][:30]}...")
        
        result = await controller.process_medical_query(
            case['query'], 
            case['patient']
        )
        
        if result["status"] == "success":
            complexity = result['complexity_analysis']['complexity_level']
            decision = result['consensus_result']['final_decision']
            print(f"   ✅ 成功 - {complexity} - {decision}")
        else:
            print(f"   ❌ 失败 - {result.get('error', '未知错误')}")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 快速入门示例")
    print("2. 多查询处理示例")
    print("3. 完整示例")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(quick_start_example())
        elif choice == "2":
            asyncio.run(multiple_queries_example())
        elif choice == "3":
            # 运行完整示例
            import sys
            sys.path.append(os.path.dirname(__file__))
            from mdagents_usage_examples import run_all_examples
            asyncio.run(run_all_examples())
        else:
            print("默认运行快速入门示例...")
            asyncio.run(quick_start_example())
            
    except KeyboardInterrupt:
        print("\n👋 感谢使用MDAgents!")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()