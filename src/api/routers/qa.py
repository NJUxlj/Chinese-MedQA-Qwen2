"""
问答服务路由
提供医疗问答接口
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import time
import json
import asyncio

from schemas.qa import QuestionRequest, QuestionResponse, StreamQuestionRequest
from providers.llm_provider import LLMProvider
from config.settings import settings

router = APIRouter()

def _build_llm_provider(model_name: str = None) -> LLMProvider:
    """从 settings.llm 配置构建 LLMProvider"""
    cfg = settings.llm
    provider = str(cfg.model_provider)

    kwargs = dict(
        provider=provider,
        model_name=model_name or str(cfg.model_name),
        base_url=str(cfg.base_url) if hasattr(cfg, 'base_url') and cfg.base_url else None,
        api_key=str(cfg.api_key) if hasattr(cfg, 'api_key') and cfg.api_key else None,
        max_tokens=int(cfg.max_tokens) if hasattr(cfg, 'max_tokens') and cfg.max_tokens else 2048,
        temperature=float(cfg.temperature) if hasattr(cfg, 'temperature') and cfg.temperature else 0.7,
        top_p=float(cfg.top_p) if hasattr(cfg, 'top_p') and cfg.top_p else 0.9,
        timeout=int(cfg.timeout) if hasattr(cfg, 'timeout') and cfg.timeout else 60,
    )

    # local 模式需要 model_path
    if provider == "local":
        model_path = str(cfg.model_path) if hasattr(cfg, 'model_path') and cfg.model_path else ""
        local_backend = str(cfg.local_backend) if hasattr(cfg, 'local_backend') and cfg.local_backend else "transformers"
        kwargs.update(model_path=model_path, local_backend=local_backend)

    return LLMProvider(**kwargs)

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    回答医疗问题

    Args:
        request: 问题请求

    Returns:
        回答结果
    """
    start_time = time.time()

    try:
        model = _build_llm_provider(request.model_name)

        # 准备提示词
        if request.use_template:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的医疗助手，请基于可靠的医学知识回答用户的问题。"
                        "请提供准确、清晰的回答，并在必要时说明信息来源或建议就医。"
                    ),
                },
                {"role": "user", "content": request.question},
            ]
        else:
            prompt = request.question

        # 生成回答
        answer = model.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        process_time = time.time() - start_time

        # 使用 tokenizer 精确计数
        try:
            tokenizer = model.tokenizer
            tokens_used = len(tokenizer.encode(answer, add_special_tokens=True))
        except Exception:
            tokens_used = len(answer)

        return QuestionResponse(
            question=request.question,
            answer=answer,
            model=request.model_name or "default",
            process_time=process_time,
            tokens_used=tokens_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask_stream")
async def ask_question_stream(request: StreamQuestionRequest):
    """
    流式回答医疗问题

    Args:
        request: 流式问题请求

    Returns:
        流式回答结果
    """
    async def event_generator():
        start_time = time.time()

        try:
            model = _build_llm_provider(request.model_name)

            if request.use_template:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业的医疗助手，请基于可靠的医学知识回答用户的问题。"
                            "请提供准确、清晰的回答，并在必要时说明信息来源或建议就医。"
                        ),
                    },
                    {"role": "user", "content": request.question},
                ]
            else:
                prompt = request.question

            full_answer = ""

            for chunk in model.generate_streaming(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                full_answer += chunk

                data = {
                    "chunk": chunk,
                    "full": full_answer,
                    "finished": False
                }

                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            process_time = time.time() - start_time
            try:
                tokenizer = model.tokenizer
                tokens_used = len(tokenizer.encode(full_answer, add_special_tokens=True))
            except Exception:
                tokens_used = len(full_answer)

            data = {
                "chunk": "",
                "full": full_answer,
                "finished": True,
                "process_time": process_time,
                "tokens_used": tokens_used
            }

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        except Exception as e:
            data = {
                "error": str(e),
                "finished": True
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@router.get("/models", response_model=List[str])
async def get_available_models():
    """
    获取可用模型列表（当前返回配置的默认模型）

    Returns:
        模型名称列表
    """
    cfg = settings.llm
    return [str(cfg.model_name)]
