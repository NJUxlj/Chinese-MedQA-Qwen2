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
from agent.tools.search_tool import SearchTool, BingSearchTool

import pytest


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


def test_search_tool_without_key():
    tool = SearchTool()
    result = tool.run(query="糖尿病治疗", num_results=3)
    assert "无法执行搜索" in result


def test_bing_search_tool_without_key():
    tool = BingSearchTool()
    result = tool.run(query="高血压指南", num_results=3)
    assert "无法执行搜索" in result


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

    print("开始测试 test_search_tool_without_key")
    test_search_tool_without_key()
    print("结束测试 test_search_tool_without_key\n")

    print("开始测试 test_bing_search_tool_without_key")
    test_bing_search_tool_without_key()
    print("结束测试 test_bing_search_tool_without_key\n")


if __name__ == "__main__":
    main()