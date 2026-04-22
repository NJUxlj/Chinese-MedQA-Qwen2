import sys
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from providers import LLMProvider


class QACorrector:
    """
    医患对话修正器

    接收当前轮次 + 历史对话 + 循证验证反馈，使用 LLM 对不符合医学事实的内容进行修正。
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def correct(
        self,
        history: List[Dict[str, str]],
        current_turn: Dict[str, str],
        feedback: str,
    ) -> Dict[str, str]:
        """
        修正当前轮次对话

        Args:
            history: 历史对话轮次列表 [{"role": ..., "content": ...}]
            current_turn: 当前轮次 {"role": ..., "content": ...}
            feedback: 循证验证的反馈意见

        Returns:
            修正后的当前轮次字典
        """
        history_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in history
        )
        prompt = f"""你是一名资深医疗质控专家，请根据以下信息修正本轮对话内容。

## 历史对话
{history_text}

## 当前轮次（待修正）
{current_turn['role']}: {current_turn['content']}

## 验证反馈（需要修正的问题）
{feedback}

## 要求
1. 仅修正与医学事实不符之处，保留对话风格。
2. 只输出修正后的内容文本，不要包含角色前缀。
3. 修正内容应简洁、准确，符合临床规范。

请输出修正后的对话内容："""

        corrected_content = self.llm.generate(prompt)
        return {"role": current_turn["role"], "content": corrected_content.strip()}
