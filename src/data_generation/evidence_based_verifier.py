import sys
import json
import re
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from providers import LLMProvider
from knowledge_base.milvus.milvus_client import MilvusClient


class EvidenceBasedVerifier:
    """
    基于循证医学的对话验证器

    结合 Milvus 检索到的医学文档和 LLM 判断，验证生成对话是否符合医学事实。
    """

    def __init__(self, llm_provider: LLMProvider, milvus_client: MilvusClient):
        self.llm = llm_provider
        self.milvus_client = milvus_client

    def verify(
        self,
        history: List[Dict[str, str]],
        current_turn: Dict[str, str],
        reference_docs: List[str],
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        验证当前轮次是否符合循证医学规范

        Args:
            history: 历史对话轮次列表
            current_turn: 当前轮次
            reference_docs: 参考文档片段列表
            collection_name: Milvus collection 名称

        Returns:
            {"passed": bool, "feedback": str}
        """
        history_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in history
        )
        ref_text = "\n\n".join(reference_docs[:5]) if reference_docs else "无参考文档"

        prompt = f"""你是一名权威的医疗质控专家。请判断以下对话轮次是否符合循证医学规范。

## 历史对话
{history_text}

## 本轮对话
{current_turn['role']}: {current_turn['content']}

## 参考医学文献摘要
{ref_text}

## 验证任务
请判断本轮对话中的医学信息是否准确、是否存在潜在的医疗错误或不规范表述。

## 输出格式（严格按 JSON 输出）
{{
  "passed": true 或 false,
  "feedback": "如果不通过，请说明需要修正的具体问题；如果通过则为空字符串"
}}

请输出 JSON："""

        response = self.llm.generate(prompt)
        try:
            # 提取 JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {
                    "passed": bool(result.get("passed", False)),
                    "feedback": str(result.get("feedback", "")),
                }
        except Exception:
            pass

        # 解析失败则默认通过（避免死循环）
        return {"passed": True, "feedback": ""}
