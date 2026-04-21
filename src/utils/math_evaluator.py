import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from utils.logger import setup_logger
from utils.metrics import DPOMetrics
from utils.base_evaluator import BaseEvaluator, EvaluatorDataset
from providers import LLMProvider


class MathEvaluator(BaseEvaluator):
    def __init__(self, config=None):
        super().__init__(config)
        self._use_api = bool(self.config.get("_use_vllm_api", False))
        if self._use_api:
            self.model = None
            self.tokenizer = None
            self._llm_provider = LLMProvider(
                provider="vllm",
                model_name=str(self.config.get("vllm_model_name", "")),
                base_url=str(self.config.get("vllm_base_url", "")),
                api_key=str(self.config.get("vllm_api_key", "")),
                max_tokens=int(self.config.get("max_new_tokens", 1024)),
                temperature=float(self.config.get("temperature", 0.0)),
            )
        else:
            self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        model_path = str(self.config.get("model_name_or_path", ""))
        device = str(self.config.get("device", "cpu"))
        padding_side = str(self.config.get("padding_side", "left"))
        use_fast = bool(self.config.get("use_fast", True))

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            use_fast=use_fast,
        )

    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        if self._use_api:
            return self._evaluate_one_sample_api(sample)
        return self.evaluate_batch_samples([sample])

    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if self._use_api:
            rows = [self._evaluate_one_sample_api(s) for s in samples]
            tc = sum(int(r.get("correct_predictions", 0)) for r in rows)
            n = len(rows)
            acc = tc / n if n else 0.0
            return {
                "math_accuracy": float(acc),
                "accuracy": float(acc),
                "exact_match": float(acc),
                "total_samples": int(n),
                "correct_predictions": int(tc),
            }

        prompt_key = str(self.config.get("prompt_key", "prompt"))
        ground_true_answer_key = str(self.config.get("ground_true_answer_key", "answer"))
        max_length = int(self.config.get("max_length", 4096))
        max_new_tokens = int(self.config.get("max_new_tokens", 1024))
        temperature = float(self.config.get("temperature", 0.0))
        enable_thinking = bool(self.config.get("enable_thinking", False))
        padding_side = str(self.config.get("padding_side", "left"))

        prompts = [s.get(prompt_key, "") for s in samples]
        messages = [{"role": "user", "content": p} for p in prompts]
        wrapped_prompts = [
            self.tokenizer.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            for m in messages
        ]

        inputs = self.tokenizer(
            wrapped_prompts,
            return_tensors="pt",
            padding=True,
            padding_side=padding_side,
            truncation=True,
            max_length=max_length,
        )

        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        pred_ids = outputs[:, input_ids.shape[1]:]
        decoded_preds = self.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        label_texts = [str(s.get(ground_true_answer_key, "")) for s in samples]

        exact_matches = 0
        for pred_text, label_text in zip(decoded_preds, label_texts):
            pred_answer = self._extract_final_answer_from_text(pred_text)
            label_answer = self._extract_final_answer_from_text(label_text)
            if self._is_answer_correct_logic(pred_answer, label_answer):
                exact_matches += 1

        total_samples = len(samples)
        accuracy = exact_matches / total_samples if total_samples > 0 else 0.0

        return {
            "math_accuracy": float(accuracy),
            "accuracy": float(accuracy),
            "exact_match": float(accuracy),
            "total_samples": int(total_samples),
            "correct_predictions": int(exact_matches)
        }

    def _evaluate_one_sample_api(self, sample: Dict[str, Any]) -> Dict[str, float]:
        prompt_key = str(self.config.get("prompt_key", "prompt"))
        ground_true_answer_key = str(self.config.get("ground_true_answer_key", "answer"))
        prompt = sample.get(prompt_key, "")
        label_raw = sample.get(ground_true_answer_key, "")
        gen_text = self._llm_provider.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=int(self.config.get("max_new_tokens", 1024)),
            temperature=float(self.config.get("temperature", 0.0)),
        )
        pred_answer = self._extract_final_answer_from_text(gen_text)
        std_answer = self._extract_final_answer_from_text(str(label_raw))
        ok = self._is_answer_correct_logic(pred_answer, std_answer)
        return {
            "math_accuracy": 1.0 if ok else 0.0,
            "accuracy": 1.0 if ok else 0.0,
            "exact_match": 1.0 if ok else 0.0,
            "total_samples": 1,
            "correct_predictions": float(ok),
        }

    def _is_answer_correct_logic(self, predicted_answer: str, standard_answer: str):
        import re
        if not predicted_answer or not standard_answer:
            return False
        predicted_answer = predicted_answer.strip()
        standard_answer = standard_answer.strip()
        if predicted_answer.lower() == standard_answer.lower():
            return True
        try:
            pred_nums = re.findall(r'[0-9]+\.?[0-9]*', predicted_answer)
            std_nums = re.findall(r'[0-9]+\.?[0-9]*', standard_answer)
            if pred_nums and std_nums:
                pred_val = float(pred_nums[0])
                std_val = float(std_nums[0])
                if abs(pred_val - std_val) < 1e-6:
                    return True
        except (ValueError, IndexError):
            pass
        try:
            if '/' in predicted_answer and '/' in standard_answer:
                pred_parts = predicted_answer.split('/')
                std_parts = standard_answer.split('/')
                if len(pred_parts) == 2 and len(std_parts) == 2:
                    pred_val = float(pred_parts[0]) / float(pred_parts[1])
                    std_val = float(std_parts[0]) / float(std_parts[1])
                    if abs(pred_val - std_val) < 1e-6:
                        return True
        except (ValueError, ZeroDivisionError):
            pass
        cleaned_pred = re.sub(r'[^\w]', '', predicted_answer.lower())
        cleaned_std = re.sub(r'[^\w]', '', standard_answer.lower())
        if cleaned_pred == cleaned_std:
            return True
        return False

    def _extract_final_answer_from_text(self, text: str):
        import re
        if not text:
            return ""
        text = text.strip()
        patterns = [
            r'答案是?\s*[:：]?\s*([^\n\r]*)',
            r'答案\s*[:：]?\s*([^\n\r]*)',
            r'答[:：]\s*([^\n\r]*)',
            r'最终答案\s*[:：]?\s*([^\n\r]*)',
            r'答案\s*\(([^)]*)\)',
            r'最终答案\s*\(([^)]*)\)',
            r'答案是?\s*([0-9]+\.?[0-9]*\s*[+\-*/]\s*[0-9]+\.?[0-9]*)',
            r'答案是?\s*([0-9]+\.?[0-9]*)',
            r'答案\s*([0-9]+\.?[0-9]*)',
            r'(?i)answer\s*[:：]?\s*([^\n\r]*)',
            r'(?i)the answer\s*[:：]?\s*([^\n\r]*)',
            r'(?i)final answer\s*[:：]?\s*([^\n\r]*)',
            r'[负-]\s*([0-9]+\.?[0-9]*)',
            r'负\s*数\s*[:：]?\s*([^\n\r]*)',
            r'([0-9]+\.?[0-9]*)\s*%',
            r'([0-9]+\.?[0-9]*)\s*个百分点',
            r'([0-9]+\.?[0-9]*)\s*百分',
            r'([0-9]+\.?[0-9]*[eE][+\-]?[0-9]+)',
            r'([0-9]+\.?[0-9]*)\s*/\s*([0-9]+\.?[0-9]*)',
            r'二分之一|三分之一|四分之一|五分之一|六分之一|七分之一|八分之一|九分之一|十分之一',
            r'([一二三四五六七八九十]+)\s*分之\s*([一二三四五六七八九十]+)',
            r'([0-9]+\.?[0-9]*)\s*[个只条件件元角分厘毫克公斤吨米厘米毫米千米公里升毫升年天日月时分秒]',
            r'([0-9]+\.?[0-9]*)\s*\+\s*([0-9]+\.?[0-9]*)\s*i',
            r'([0-9]+\.?[0-9]*)\s*[+\-]\s*([0-9]+\.?[0-9]*)\s*[jJ]',
            r'√\s*([0-9]+\.?[0-9]*)',
            r'根号\s*([0-9]+\.?[0-9]*)',
            r'√\s*\(\s*([0-9]+\.?[0-9]*)\s*\)',
            r'\b[IVX]+\b',
            r'\(\s*([^\)]*)\s*\)',
            r'【\s*([^\]]*)\s*】',
            r'《\s*([^\>]*)\s*》',
            r'"([^"]*)"',
            r"'([^']*)'",
            r'≈\s*([0-9]+\.?[0-9]*)',
            r'约\s*([0-9]+\.?[0-9]*)',
            r'大约\s*([0-9]+\.?[0-9]*)',
            r'大概\s*([0-9]+\.?[0-9]*)',
            r'[零一二三四五六七八九十百千万亿]+',
            r'[0-9]+\.?[0-9]*\s*[、，,\s]*\s*[0-9]+\.?[0-9]*\s*[、，,\s]*\s*[0-9]+\.?[0-9]*',
            r'([0-9]+\.?[0-9]*)\s*[:：]\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*比\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*\^\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*\*\*\s*([0-9]+\.?[0-9]*)',
            r'(?i)equals?\s*[:：]?\s*([^\n\r]*)',
            r'(?i)result\s*[:：]?\s*([^\n\r]*)',
            r'(?i)solution\s*[:：]?\s*([^\n\r]*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if match.lastindex and match.lastindex >= 1:
                    answer = match.group(1).strip()
                else:
                    answer = match.group(0).strip()
                answer = re.sub(r'[。！？.,，；：\s]+$', '', answer)
                if answer:
                    return answer
        numbers = re.findall(r'[0-9]+\.?[0-9]*', text)
        if numbers:
            return numbers[-1]
        return ""
