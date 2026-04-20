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
from services.model_service import get_model_service, ModelService

router = APIRouter()

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    回答医疗问题
    
    Args:
        request: 问题请求
        model_service: 模型服务
    
    Returns:
        回答结果
    """
    start_time = time.time()
    
    try:
        # 获取模型
        model = model_service.get_model(request.model_name)
        
        # 准备提示词
        if request.use_template:
            # 使用系统提示词模板
            prompt = f"""<|im_start|>system
你是一个专业的医疗助手，请基于可靠的医学知识回答用户的问题。
请提供准确、清晰的回答，并在必要时说明信息来源或建议就医。
<|im_end|>
<|im_start|>user
{request.question}
<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = request.question
        
        # 生成回答
        answer = model.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 处理结果
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
async def ask_question_stream(
    request: StreamQuestionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    流式回答医疗问题
    
    Args:
        request: 流式问题请求
        model_service: 模型服务
    
    Returns:
        流式回答结果
    """
    async def event_generator():
        start_time = time.time()
        
        try:
            # 获取模型
            model = model_service.get_model(request.model_name)
            
            # 准备提示词
            if request.use_template:
                # 使用系统提示词模板
                prompt = f"""<|im_start|>system
你是一个专业的医疗助手，请基于可靠的医学知识回答用户的问题。
请提供准确、清晰的回答，并在必要时说明信息来源或建议就医。
<|im_end|>
<|im_start|>user
{request.question}
<|im_end|>
<|im_start|>assistant
"""
            else:
                prompt = request.question
            
            # 全部答案
            full_answer = ""
            
            # 流式生成
            for chunk in model.generate_streaming(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                full_answer += chunk
                
                # 创建SSE事件
                data = {
                    "chunk": chunk,
                    "full": full_answer,
                    "finished": False
                }
                
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # 小延迟避免过快
            
            # 发送完成事件
            process_time = time.time() - start_time
            # 使用 tokenizer 精确计数
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
            # 发送错误事件
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
async def get_available_models(
    model_service: ModelService = Depends(get_model_service)
):
    """
    获取可用模型列表
    
    Args:
        model_service: 模型服务
    
    Returns:
        模型名称列表
    """
    return model_service.get_loaded_models()