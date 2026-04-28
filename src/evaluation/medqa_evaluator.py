import json
import re
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
from typing import Dict, Any, List

from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider
from concurrent.futures import as_completed, ThreadPoolExecutor



class MedQAEvaluator(BaseEvaluator):
    """使用 LLM 作为裁判，判断模型输出是否与标准答案一致。

    支持语义匹配，允许同义词替换。
    支持带思考能力的推理模型（如 MiniMax-M2.7），通过 <answer> 标签解析最终结果。
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.question_key = str(self.config.get("question_key", "question"))
        self.model_answer_key = str(self.config.get("model_answer_key", "output"))
        self.ground_true_answer_key = str(self.config.get("ground_true_answer_key", "answer"))

        self.logger.info("MedQAEvaluator 初始化完成")

        self.llm_provider = LLMProvider(
            provider="vllm",
            model_name=str(self.config.get("model_name", "")),
            base_url=str(self.config.get("base_url", "")),
            api_key=str(self.config.get("api_key", "")),
            max_tokens=int(self.config.get("max_tokens", 2048)),
            temperature=float(self.config.get("temperature", 0.7)),
        )

        self.logger.info(f"MedQAEvaluator 初始化完成，模型名称: {self.config.get('model_name', '')}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        prompt = f"""你是一个拥有丰富临床经验的医疗专家，请判断模型回答是否与标准答案一致（语义相同即可，不要求字面完全一致）。

## 用户问题
{sample.get(self.question_key, "")}

## 模型回答
{sample.get(self.model_answer_key, "")}

## 标准答案
{sample.get(self.ground_true_answer_key, '')}

## 判断规则
- 语义相同即可通过，允许同义词替换（如"体重减轻"="体重下降"、"mmHg"="毫米汞柱"）
- 允许表达方式不同（如"或"="和/或"）
- 允许轻微的表述差异
- 数值必须等价

## 输出要求
请在 <answer>True</answer> 或 <answer>False</answer> 标签中输出最终判断结果。
例如：<answer>True</answer>

请开始判断："""
        return prompt

    def _extract_answer(self, response: str) -> str:
        """从响应中提取 <answer>True/False</answer> 标签内的结果。"""
        if not response:
            return "False"
        match = re.search(r"<answer>\s*(True|False)\s*</answer>", response, re.IGNORECASE)
        if match:
            return match.group(1)
        stripped = response.strip().lower()
        if stripped == "true":
            return "True"
        if stripped == "false":
            return "False"
        self.logger.warning(f"无法从响应中提取答案标签，返回 False。响应内容: {response[:200]}")
        return "False"

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.format_prompt(sample)
        response = self.llm_provider.generate(prompt)
        verdict = self._extract_answer(response)
        is_correct = verdict.lower() == "true"
        return {
            "is_correct": is_correct,
            "llm_verdict": verdict,
        }

    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.evaluate_one_sample(s) for s in samples]
        return {
            "results": results,
            "accuracy": float(np.mean([r["is_correct"] for r in results])) if results else 0.0,
        }


    def evaluate(self) -> Dict[str, float]:

        # load_dataset
        self.test_dataset = self.load_test_dataset(self.config.get("dataset_path", ""))
        results = []
        # evaluate
        with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 1)) as executor:
            future_to_sample = {executor.submit(self.evaluate_one_sample, s): s for s in self.test_dataset}
            with tqdm(total=len(self.test_dataset), desc="Evaluating", position=1, leave=False) as pbar:
                for future in as_completed(future_to_sample):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"样本 {future_to_sample[future]} 处理失败: {e}")
                        raise

        # calculate accuracy
        accuracy = float(np.mean([r["is_correct"] for r in results])) if results else 0.0
        self.logger.info(f"在测试数据集 [{self.config.get('dataset_path', '')}] 上，模型的准确率为: {accuracy}")

        return {
            "accuracy": accuracy,
        }
