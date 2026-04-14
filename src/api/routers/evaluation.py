"""
评估服务路由
提供模型和系统评估接口
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from services.model_service import get_model_service, ModelService
from services.rag_service import get_rag_service, RAGService
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__, level="INFO")

class EvaluationRequest(BaseModel):
    """评估请求"""
    question: str
    reference_answer: str
    model_name: Optional[str] = None
    use_rag: bool = False
    kb_name: Optional[str] = None
    metrics: List[str] = ["similarity", "f1", "rouge"]

class EvaluationResponse(BaseModel):
    """评估响应"""
    question: str
    reference_answer: str
    model_answer: str
    metrics: Dict[str, float]
    model_name: str
    process_time: float
    additional_info: Optional[Dict[str, Any]] = None

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    request: EvaluationRequest,
    model_service: ModelService = Depends(get_model_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    评估模型回答质量
    
    Args:
        request: 评估请求
        model_service: 模型服务
        rag_service: RAG服务
    
    Returns:
        评估结果
    """
    import time
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.feature_extraction.text import CountVectorizer
    
    start_time = time.time()
    
    try:
        # 获取模型回答
        model_answer = ""
        additional_info = {}
        
        if request.use_rag and request.kb_name:
            # 使用RAG
            response = rag_service.generate_response(
                kb_name=request.kb_name,
                query=request.question,
                model_name=request.model_name
            )
            model_answer = response.get("answer", "")
            additional_info["retrieved_contexts"] = response.get("contexts", [])
            additional_info["sources"] = response.get("sources", [])
        else:
            # 直接使用模型
            model = model_service.get_model(request.model_name)
            model_answer = model.generate(
                prompt=request.question,
                max_new_tokens=1024,
                temperature=0.7
            )
        
        # 计算评估指标
        metrics = {}
        
        # 简单相似度评分
        if "similarity" in request.metrics:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, request.reference_answer, model_answer).ratio()
            metrics["similarity"] = similarity
        
        # F1分数 (简化版)
        if "f1" in request.metrics:
            # 将文本转换为单词袋
            vectorizer = CountVectorizer().fit([request.reference_answer, model_answer])
            reference_vec = vectorizer.transform([request.reference_answer]).toarray()[0]
            model_vec = vectorizer.transform([model_answer]).toarray()[0]
            
            # 二值化向量
            reference_binary = np.where(reference_vec > 0, 1, 0)
            model_binary = np.where(model_vec > 0, 1, 0)
            
            f1 = f1_score(reference_binary, model_binary, average='micro')
            metrics["f1"] = f1
        
        # ROUGE分数 (需要rouge库)
        if "rouge" in request.metrics:
            try:
                from rouge import Rouge
                rouge = Rouge()
                scores = rouge.get_scores(model_answer, request.reference_answer)[0]
                
                metrics["rouge-1"] = scores["rouge-1"]["f"]
                metrics["rouge-2"] = scores["rouge-2"]["f"]
                metrics["rouge-l"] = scores["rouge-l"]["f"]
            except ImportError:
                metrics["rouge"] = "rouge库未安装"
        
        process_time = time.time() - start_time
        
        return EvaluationResponse(
            question=request.question,
            reference_answer=request.reference_answer,
            model_answer=model_answer,
            metrics=metrics,
            model_name=request.model_name or "default",
            process_time=process_time,
            additional_info=additional_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchEvaluationRequest(BaseModel):
    """批量评估请求"""
    questions: List[Dict[str, str]]  # 包含question和reference_answer
    model_name: Optional[str] = None
    use_rag: bool = False
    kb_name: Optional[str] = None
    metrics: List[str] = ["similarity", "f1", "rouge"]

class BatchEvaluationResponse(BaseModel):
    """批量评估响应"""
    results: List[EvaluationResponse]
    average_metrics: Dict[str, float]
    total_time: float

@router.post("/batch_evaluate", response_model=BatchEvaluationResponse)
async def batch_evaluate_model(
    request: BatchEvaluationRequest,
    model_service: ModelService = Depends(get_model_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    批量评估模型回答质量
    
    Args:
        request: 批量评估请求
        model_service: 模型服务
        rag_service: RAG服务
    
    Returns:
        批量评估结果
    """
    import time
    
    start_time = time.time()
    results = []
    
    for item in request.questions:
        eval_request = EvaluationRequest(
            question=item["question"],
            reference_answer=item["reference_answer"],
            model_name=request.model_name,
            use_rag=request.use_rag,
            kb_name=request.kb_name,
            metrics=request.metrics
        )
        
        try:
            result = await evaluate_model(eval_request, model_service, rag_service)
            results.append(result)
        except Exception as e:
            # 处理单个评估失败
            logger.error(f"评估失败: {e}")
    
    # 计算平均指标
    avg_metrics = {}
    
    if results:
        # 收集所有指标名称
        all_metric_names = set()
        for result in results:
            all_metric_names.update(result.metrics.keys())
        
        # 计算每个指标的平均值
        for metric_name in all_metric_names:
            values = [r.metrics.get(metric_name) for r in results if metric_name in r.metrics]
            values = [v for v in values if isinstance(v, (int, float))]
            
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
    
    total_time = time.time() - start_time
    
    return BatchEvaluationResponse(
        results=results,
        average_metrics=avg_metrics,
        total_time=total_time
    )