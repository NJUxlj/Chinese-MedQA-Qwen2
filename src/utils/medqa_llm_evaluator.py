import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
from typing import Dict, Any, List

from utils.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider


class MedQALLMEvaluator(BaseEvaluator):
    """使用 LLM 作为裁判，判断模型输出是否与标准答案一致。

    仅支持两种推理后端：（1）vLLM/OpenAI 兼容 API；（2）本地 transformers 权重。
    """

    def __init__(self, config=None, llm_config=None):
        super().__init__(config)
        self._init_judge_provider()

        self.ground_true_answer_key = str(
            self.config.get("ground_true_answer_key", "answer")
        )
        self.logger.info("MedQALLMEvaluator 初始化完成")

    def _init_judge_provider(self) -> None:
        vllm_ok = bool(self.config.get("_vllm_config_valid", False))
        max_tokens = int(self.config.get("max_tokens", 2048))
        temperature = float(self.config.get("temperature", 0.7))
        if vllm_ok:
            self.llm_provider = LLMProvider(
                provider="vllm",
                model_name=str(self.config.get("vllm_model_name", "")),
                base_url=str(self.config.get("vllm_base_url", "")),
                api_key=str(self.config.get("vllm_api_key", "")),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            model_path = str(self.config.get("model_name_or_path", "") or "").strip()
            if not model_path or model_path == "__vllm_only__":
                raise ValueError(
                    "MedQA LLM 评估：vLLM 配置无效时必须提供有效的本地 model_name_or_path"
                )
            self.llm_provider = LLMProvider(
                provider="local",
                model_path=model_path,
                local_backend="transformers",
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        system_prompt = f"""
## 角色
你是一个拥有丰富临床经验的医疗专家，你的任务是根据用户的问题和上下文，判断模型的回答是否与标准答案一致。

## 任务：
请根据用户问题，判断模型回答是否与标准答案一致。

## 规则:
- 模型回答必须与标准答案完全一致，包括大小写、标点符号等。
- 如果模型回答中包含多个选项，必须全部选择正确。
- 如果模型回答中包含数值，必须与标准答案完全一致。

## 用户问题
{sample.get('question', sample.get('input', ''))}

## 模型回答
{sample.get('output', '')}

## 标准答案
{sample.get(self.ground_true_answer_key, '')}

## 输出格式
你只能输出 True 或 False， 除此以外不能输出任何东西。

## 请你开始判断

"""
        return system_prompt

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.format_prompt(sample)
        response = self.llm_provider.generate(prompt)
        verdict = response.strip() if response else "False"
        is_correct = verdict.lower() == "true"
        return {
            "is_correct": is_correct,
            "llm_verdict": verdict,
        }

    def evaluate_one_sample_by_api(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.evaluate_one_sample(sample)

    def evaluate_batch_samples_by_api(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.evaluate_one_sample_by_api(s) for s in samples]

    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.evaluate_one_sample(s) for s in samples]
        return {
            "results": results,
            "accuracy": float(np.mean([r["is_correct"] for r in results])) if results else 0.0,
        }
