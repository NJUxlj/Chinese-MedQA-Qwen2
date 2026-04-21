import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import evaluate

from typing import Dict, List, Any

from config.settings import settings
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset


class BleuRougeEvaluator(BaseEvaluator):
    """基于 BLEU / ROUGE / Perplexity 的评估器。

    所有配置均从 settings.evaluator 读取，无需单独的配置文件。
    """

    def __init__(self, config=None):
        """
        Args:
            config: 可选配置对象（omegaconf DictConfig）。
                    若为 None，则使用 settings.evaluator。
        """
        super().__init__(config)

        model_path = str(self.config.model_name_or_path)
        device = str(getattr(self.config, "device", "cpu"))

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

        self.generation_config = {
            "max_new_tokens": int(getattr(self.config, "max_new_tokens", 2048)),
            "temperature": float(getattr(self.config, "temperature", 0.7)),
            "top_p": float(getattr(self.config, "top_p", 0.9)),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        self.prompt_key = str(getattr(self.config, "prompt_key", "prompt"))
        self.ground_true_answer_key = str(getattr(self.config, "ground_true_answer_key", "answer"))
        self.device = device

        self.logger.info(f"BLEU-ROUGE评估器初始化完成，模型路径: {model_path}")

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """评估单个样本（委托给 evaluate_batch_sample）"""
        batch_result = self.evaluate_batch_sample([sample])
        return {
            k: (v[0] if isinstance(v, list) and len(v) > 0 else v)
            for k, v in batch_result.items()
        }

    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估样本"""
        results: Dict[str, Any] = {
            "perplexity": [],
            "bleu": [],
            "rouge": [],
        }

        prompts = [sample[self.prompt_key] for sample in samples]
        ground_true_answers = [sample[self.ground_true_answer_key] for sample in samples]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_config)
            predictions = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            loss = self.model(
                inputs["input_ids"],
                labels=inputs["input_ids"],
            ).loss
            perplexity = torch.exp(loss).item()

        results["perplexity"].append(perplexity)

        results["bleu"].extend([
            self.bleu.compute(predictions=[p], references=[r])["bleu"]
            for p, r in zip(predictions, ground_true_answers)
        ])

        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=ground_true_answers,
            rouge_types=["rougeL"],
        )
        rougeL = rouge_scores.get("rougeL", 0.0)
        if isinstance(rougeL, list):
            results["rouge"].extend(rougeL)
        else:
            results["rouge"].extend([rougeL] * len(predictions))

        return results

    def evaluate(self) -> Dict[str, float]:
        """遍历全部测试集评估"""
        results: Dict[str, Any] = {
            "perplexity": [],
            "bleu": [],
            "rouge": [],
        }

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="BLEU/ROUGE Evaluating"):
                if isinstance(batch, dict):
                    batch = [batch]
                batch_results = self.evaluate_batch_sample(batch)
                results["perplexity"].extend(batch_results["perplexity"])
                results["bleu"].extend(batch_results["bleu"])
                results["rouge"].extend(batch_results["rouge"])

        return {
            "perplexity": float(np.mean(results["perplexity"])) if results["perplexity"] else 0.0,
            "bleu": float(np.mean(results["bleu"])) if results["bleu"] else 0.0,
            "rougeL": float(np.mean(results["rouge"])) if results["rouge"] else 0.0,
        }

    def save_results(self, results: Dict[str, float]):
        """保存评估结果"""
        save_path = str(getattr(self.config, "eval_result_save_path", "./eval_results.json"))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def run():
    """运行评估器（使用 settings.evaluator 配置）"""
    evaluator = BleuRougeEvaluator()
    results = evaluator.evaluate()
    evaluator.save_results(results)
    print(f"评估完成！结果: {results}")


if __name__ == "__main__":
    run()
