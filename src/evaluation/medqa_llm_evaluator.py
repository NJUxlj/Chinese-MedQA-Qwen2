import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
from typing import Dict, Any, List

from config.settings import settings
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider


class MedQALLMEvaluator(BaseEvaluator):
    """使用外部 LLM 作为裁判，判断模型输出是否与标准答案一致。

    所有配置均从 settings.evaluator（测试集路径/批大小等）
    和 settings.llm（LLM 调用参数）读取，无需单独配置文件。
    """

    def __init__(self, config=None, llm_config=None):
        """
        Args:
            config: 评估器配置（默认 settings.evaluator）
            llm_config: LLM 配置（默认 settings.llm）
        """
        super().__init__(config)
        self.llm_config = llm_config if llm_config is not None else settings.llm

        self.llm_provider = LLMProvider(
            provider=str(self.llm_config.model_provider),
            model_name=str(self.llm_config.model_name),
            base_url=str(self.llm_config.base_url),
            api_key=str(self.llm_config.api_key),
            max_tokens=int(getattr(self.llm_config, "max_tokens", 2048)),
            temperature=float(getattr(self.llm_config, "temperature", 0.7)),
        )

        self.ground_true_answer_key = str(
            getattr(self.config, "ground_true_answer_key", "answer")
        )
        self.logger.info("MedQALLMEvaluator 初始化完成")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """构建送给裁判 LLM 的评估 Prompt"""
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
        """对单个样本进行评估"""
        prompt = self.format_prompt(sample)
        response = self.llm_provider.generate(prompt)
        verdict = response.strip() if response else "False"
        is_correct = verdict.lower() == "true"
        return {
            "is_correct": is_correct,
            "llm_verdict": verdict,
        }

    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估"""
        results = [self.evaluate_one_sample(s) for s in samples]
        return {
            "results": results,
            "accuracy": float(np.mean([r["is_correct"] for r in results])) if results else 0.0,
        }

    def evaluate(self) -> Dict[str, Any]:
        """遍历全部测试集评估"""
        all_results = []

        for batch in tqdm(self.test_dataloader, desc="MedQA LLM Evaluating"):
            if isinstance(batch, dict):
                batch = [batch]
            batch_result = self.evaluate_batch_sample(batch)
            all_results.extend(batch_result["results"])

        accuracy = float(np.mean([r["is_correct"] for r in all_results])) if all_results else 0.0
        self.logger.info(f"评估完成，准确率: {accuracy:.4f}，共 {len(all_results)} 条样本")
        return {
            "accuracy": accuracy,
            "total_samples": len(all_results),
            "correct_samples": int(sum(r["is_correct"] for r in all_results)),
            "results": all_results,
        }


def run():
    """运行 MedQA LLM 评估器（使用 settings.evaluator 配置）"""
    evaluator = MedQALLMEvaluator()
    results = evaluator.evaluate()
    print(f"评估完成！准确率: {results['accuracy']:.4f}")
    print(f"总样本数: {results['total_samples']}，正确数: {results['correct_samples']}")


if __name__ == "__main__":
    run()
