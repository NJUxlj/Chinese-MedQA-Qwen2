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

from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider
from concurrent.futures import as_completed, ThreadPoolExecutor


class BleuRougeEvaluator(BaseEvaluator):
    """基于 BLEU / ROUGE / Perplexity 的评估器。"""

    def __init__(self, config=None):
        super().__init__(config)

        self.question_key = str(self.config.get("question_key", "question"))
        self.model_answer_key = str(self.config.get("model_answer_key", "output"))
        self.ground_true_answer_key = str(self.config.get("ground_true_answer_key", "answer"))

        self._smooth = SmoothingFunction().method1
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        self.llm_provider = LLMProvider(
            provider="vllm",
            model_name=str(self.config.get("model_name", "")),
            base_url=str(self.config.get("base_url", "")),
            api_key=str(self.config.get("api_key", "")),
            max_tokens=int(self.config.get("max_new_tokens", 2048)),
            temperature=float(self.config.get("temperature", 0.7)),
        )
        self.logger.info("BLEU-ROUGE 评估器初始化完成")


    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            import jieba
            return list(jieba.cut(text))
        except Exception:
            return text.split()

    def _bleu_single(self, prediction: str, reference: str) -> float:
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(prediction)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return float(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self._smooth)
        )

    def _rouge_l_single(self, prediction: str, reference: str) -> float:
        return float(self._rouge.score(reference, prediction)["rougeL"].fmeasure)


    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        row = self._score_bleu_rouge_single(sample)
        return {"bleu": row["bleu"], "rouge": row["rougeL"], "perplexity": row["perplexity"]}

    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._score_bleu_rouge_single(s) for s in samples]

    def _score_bleu_rouge_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample[self.question_key]
        ref = sample[self.ground_true_answer_key]
        prediction = self.llm_provider.generate(prompt)
        return {
            "bleu": self._bleu_single(prediction, ref),
            "rougeL": self._rouge_l_single(prediction, ref),
            "perplexity": 0.0,
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
        bleu = float(np.mean([r["bleu"] for r in results])) if results else 0.0
        rouge = float(np.mean([r["rouge"] for r in results])) if results else 0.0
        perplexity = float(np.mean([r["perplexity"] for r in results])) if results else 0.0
        self.logger.info(f"在测试数据集 [{self.config.get('dataset_path', '')}] 上，模型的BLEU分数为: {bleu}, ROUGE分数为: {rouge}, 困惑度为: {perplexity}")

        return {
            "bleu": bleu,
            "rouge": rouge,
            "perplexity": perplexity,
        }
