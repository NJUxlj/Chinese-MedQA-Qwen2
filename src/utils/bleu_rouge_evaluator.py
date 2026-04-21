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
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from typing import Dict, List, Any

from utils.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider


class BleuRougeEvaluator(BaseEvaluator):
    """基于 BLEU / ROUGE / Perplexity 的评估器。"""

    def __init__(self, config=None):
        super().__init__(config)

        self._use_api = bool(self.config.get("_use_vllm_api", False))

        self.prompt_key = str(self.config.get("prompt_key", "prompt"))
        self.ground_true_answer_key = str(self.config.get("ground_true_answer_key", "answer"))
        self.device = str(self.config.get("device", "cpu"))

        self._smooth = SmoothingFunction().method1
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        if self._use_api:
            self.model = None
            self.tokenizer = None
            self._llm_provider = LLMProvider(
                provider="vllm",
                model_name=str(self.config.get("vllm_model_name", "")),
                base_url=str(self.config.get("vllm_base_url", "")),
                api_key=str(self.config.get("vllm_api_key", "")),
                max_tokens=int(self.config.get("max_new_tokens", 2048)),
                temperature=float(self.config.get("temperature", 0.7)),
            )
            self.generation_config = {}
            self.logger.info("BLEU-ROUGE 评估器初始化完成（vLLM API 模式）")
        else:
            model_path = str(self.config.get("model_name_or_path", ""))
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generation_config = {
                "max_new_tokens": int(self.config.get("max_new_tokens", 2048)),
                "temperature": float(self.config.get("temperature", 0.7)),
                "top_p": float(self.config.get("top_p", 0.9)),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            self.logger.info(f"BLEU-ROUGE评估器初始化完成，模型路径: {model_path}")

    @staticmethod
    def _tokenize_for_nlg(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            import jieba
            return list(jieba.cut(text))
        except Exception:
            return text.split()

    def _bleu_single(self, prediction: str, reference: str) -> float:
        ref_t = self._tokenize_for_nlg(reference)
        hyp_t = self._tokenize_for_nlg(prediction)
        if not ref_t or not hyp_t:
            return 0.0
        return float(
            sentence_bleu([ref_t], hyp_t, smoothing_function=self._smooth)
        )

    def _rouge_l_single(self, prediction: str, reference: str) -> float:
        return float(self._rouge.score(reference, prediction)["rougeL"].fmeasure)

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        if self._use_api:
            return self.evaluate_one_sample_by_api(sample)
        batch_result = self.evaluate_batch_sample([sample])
        return {
            k: (v[0] if isinstance(v, list) and len(v) > 0 else v)
            for k, v in batch_result.items()
        }

    def evaluate_one_sample_by_api(self, sample: Dict[str, Any]) -> Dict[str, float]:
        row = self._score_bleu_rouge_api_single(sample)
        return {"bleu": row["bleu"], "rouge": row["rougeL"], "perplexity": row["perplexity"]}

    def evaluate_batch_samples_by_api(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._score_bleu_rouge_api_single(s) for s in samples]

    def _score_bleu_rouge_api_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample[self.prompt_key]
        ref = sample[self.ground_true_answer_key]
        prediction = self._llm_provider.generate(prompt)
        return {
            "bleu": self._bleu_single(prediction, ref),
            "rougeL": self._rouge_l_single(prediction, ref),
            "perplexity": 0.0,
        }

    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._use_api:
            rows = self.evaluate_batch_samples_by_api(samples)
            return {
                "perplexity": [r["perplexity"] for r in rows],
                "bleu": [r["bleu"] for r in rows],
                "rouge": [r["rougeL"] for r in rows],
            }

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

        results["bleu"].extend(
            self._bleu_single(p, r) for p, r in zip(predictions, ground_true_answers)
        )
        results["rouge"].extend(
            self._rouge_l_single(p, r) for p, r in zip(predictions, ground_true_answers)
        )

        return results
