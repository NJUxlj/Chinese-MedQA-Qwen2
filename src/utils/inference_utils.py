"""评估推理模式：仅支持本地权重或 vLLM OpenAI 兼容 API 两种模式。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_valid_vllm_triple(
    model_name: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> bool:
    """
    判断 [model_name, base_url, api_key] 是否可作为有效 vLLM/OpenAI 兼容端点。
    api_key 允许为空字符串（无鉴权部署）。
    """
    if model_name is None or not str(model_name).strip():
        return False
    if base_url is None or not str(base_url).strip():
        return False
    if api_key is None:
        return False
    return True


def normalize_dataset_rows(
    rows: List[Dict[str, Any]],
    question_key: str,
    model_answer_key: str,
    ground_true_answer_key: str,
) -> List[Dict[str, Any]]:
    """
    将数据集行规范为各评估器共用的 question / output / answer / prompt 字段，
    同时保留原始行中的其它字段（如 code 任务的 language、test_cases）。
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        q = row.get(question_key, "")
        ma = row.get(model_answer_key, "")
        gt = row.get(ground_true_answer_key, "")
        merged = dict(row)
        merged["question"] = q
        merged["output"] = ma
        merged["answer"] = gt
        merged["prompt"] = q
        merged["code"] = ma
        out.append(merged)
    return out


def display_model_name(
    *,
    vllm_valid: bool,
    model_name: Optional[str],
    model_path: str,
) -> str:
    if vllm_valid and model_name:
        return str(model_name).strip()
    return (model_path or "").strip() or "local"
