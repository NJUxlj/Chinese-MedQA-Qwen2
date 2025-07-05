import os
import sys
import re
import types

# 将项目的 src 目录加入到 PYTHONPATH，方便导入工具模块
PROJECT_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if PROJECT_SRC_PATH not in sys.path:
    sys.path.insert(0, PROJECT_SRC_PATH)

# 现在可以安全地导入工具
from agent.tools.calculator_tool import CalculatorTool
from agent.tools.medical_reference_tool import MedicalReferenceTool
from agent.tools.search_tool import SearchTool, BingSearchTool
from agent.tools.reaction_agent_tool import ReActAgentTool

import pytest


class DummyLLM:
    """简易语言模型桩，用于测试 ReActAgentTool"""

    def generate(self, messages, temperature: float = 0.7, max_tokens: int = 1000):
        # 简单返回固定格式，包含结论，满足工具的解析需求
        return (
            "思考: 这是一次单步推理\n"
            "行动: 无需调用工具\n"
            "结论: 测试完成"
        )


def test_calculator_basic():
    tool = CalculatorTool()
    result = tool.run(expression="2 + 2")
    assert "计算结果" in result
    assert re.search(r"[=:：]\s*4(\.0+)?", result)


def test_calculator_dose():
    tool = CalculatorTool()
    result = tool.run(expression="5 mg/kg * 10 kg")
    assert "剂量计算结果" in result
    assert "50" in result


def test_medical_reference_exact():
    tool = MedicalReferenceTool()
    result = tool.run(query_type="诊断标准", query_term="糖尿病")
    assert "糖尿病" in result
    assert "空腹血糖" in result


def test_medical_reference_not_found():
    tool = MedicalReferenceTool()
    result = tool.run(query_type="药物剂量", query_term="不存在的药物")
    assert "未找到" in result


def test_medical_reference_bad_type():
    tool = MedicalReferenceTool()
    result = tool.run(query_type="哈哈哈", query_term="糖尿病")
    assert "错误" in result


def test_search_tool_without_key():
    tool = SearchTool()
    result = tool.run(query="糖尿病治疗", num_results=3)
    assert "无法执行搜索" in result


def test_bing_search_tool_without_key():
    tool = BingSearchTool()
    result = tool.run(query="高血压指南", num_results=3)
    assert "无法执行搜索" in result


def test_react_agent_simple():
    dummy_model = DummyLLM()
    tool = ReActAgentTool(model=dummy_model, available_tools={})
    output = tool.run(task="测试任务", max_steps=3)
    assert "最终结论" in output or "未得出明确结论" in output
    assert "测试完成" in output 









def main():
    '''
    1. 测试工具
    2. 测试工具的输出
    3. 测试工具的输入
    4. 测试工具的错误处理
    5. 测试工具的性能
    6. 测试工具的并发处理


    python src/tests/test_tools.py
    '''
    
    print("开始测试 test_calculator_basic")
    test_calculator_basic()
    print("结束测试 test_calculator_basic\n")

    print("开始测试 test_calculator_dose")
    test_calculator_dose()
    print("结束测试 test_calculator_dose\n")

    print("开始测试 test_medical_reference_exact")
    test_medical_reference_exact()
    print("结束测试 test_medical_reference_exact\n")

    print("开始测试 test_medical_reference_not_found")
    test_medical_reference_not_found()
    print("结束测试 test_medical_reference_not_found\n")

    print("开始测试 test_medical_reference_bad_type")
    try:
        test_medical_reference_bad_type()
    except Exception as e:
        print(str(e))
    print("结束测试 test_medical_reference_bad_type\n")

    print("开始测试 test_search_tool_without_key")
    test_search_tool_without_key()
    print("结束测试 test_search_tool_without_key\n")

    print("开始测试 test_bing_search_tool_without_key")
    test_bing_search_tool_without_key()
    print("结束测试 test_bing_search_tool_without_key\n")









if __name__ == "__main__":
    main()