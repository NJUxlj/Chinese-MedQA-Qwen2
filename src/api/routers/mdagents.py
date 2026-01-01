"""
MDAgents API路由
提供医疗多智能体系统的RESTful API接口
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import Field

from api.schemas.mdagents import (
    MedicalQueryRequest, MedicalQueryResponse,
    ComplexityAnalysisRequest, ComplexityAnalysisResponse,
    CollaborationStrategyRequest, CollaborationStrategyResponse,
    AgentConsultationRequest, AgentConsultationResponse,
    ReviewRequest, ReviewResultResponse,
    AgentStatusResponse, SystemStatusResponse,
    HealthResponse, ErrorResponse
)
from api.services.mdagents_service import get_mdagents_service, MDAgentsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mdagents", tags=["MDAgents医疗多智能体系统"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查MDAgents服务健康状态"""
    service = get_mdagents_service()
    return service.health_check()


@router.get("/status", response_model=SystemStatusResponse, summary="获取系统状态")
async def get_system_status():
    """获取MDAgents系统运行状态"""
    service = get_mdagents_service()
    return service.get_system_status()


@router.post("/query", response_model=MedicalQueryResponse, summary="处理医疗查询")
async def process_medical_query(request: MedicalQueryRequest):
    """
    处理用户医疗查询
    
    - **query**: 医疗问题描述（必填）
    - **patient_context**: 患者背景信息（可选）
    - **options**: 其他配置选项（可选）
    """
    service = get_mdagents_service()
    return await service.process_medical_query(request)


@router.post("/complexity/analyze", response_model=ComplexityAnalysisResponse, summary="分析查询复杂度")
async def analyze_complexity(request: ComplexityAnalysisRequest):
    """
    分析医疗查询的复杂度
    
    - **query**: 待分析的医疗查询（必填）
    - **include_features**: 是否返回详细特征（默认True）
    """
    service = get_mdagents_service()
    return await service.analyze_complexity(request)


@router.post("/collaboration/strategy", response_model=CollaborationStrategyResponse, summary="获取协作策略")
async def get_collaboration_strategy(request: CollaborationStrategyRequest):
    """
    根据复杂度等级获取推荐的协作策略
    
    - **complexity_level**: 查询复杂度等级（必填）
    - **query**: 医疗查询内容（必填）
    - **context**: 额外上下文信息（可选）
    """
    service = get_mdagents_service()
    return await service.get_collaboration_strategy(request)


@router.post("/consultation", response_model=AgentConsultationResponse, summary="智能体会诊")
async def process_consultation(request: AgentConsultationRequest):
    """
    处理多智能体会诊请求
    
    - **query**: 会诊问题（必填）
    - **agent_types**: 参与的智能体类型列表（必填）
    - **consensus_method**: 共识方法（可选）
    """
    service = get_mdagents_service()
    return await service.process_agent_consultation(request)


@router.post("/review", response_model=ReviewResultResponse, summary="审查医疗建议")
async def process_review(request: ReviewRequest):
    """
    审查医疗建议的质量和风险
    
    - **medical_advice**: 待审查的医疗建议（必填）
    - **agent_recommendations**: 智能体推荐列表（可选）
    - **complexity_level**: 复杂度等级（可选）
    """
    service = get_mdagents_service()
    return await service.process_review(request)


@router.get("/agents", response_model=list[AgentStatusResponse], summary="获取所有智能体状态")
async def get_all_agents_status():
    """获取所有智能体的当前状态"""
    service = get_mdagents_service()
    return service.get_all_agent_status()


@router.get("/agents/{agent_type}", response_model=AgentStatusResponse, summary="获取指定智能体状态")
async def get_agent_status(agent_type: str):
    """
    获取指定类型智能体的状态
    
    - **agent_type**: 智能体类型（pcc, specialist, moderator, recruiter, reviewer, integrator）
    """
    service = get_mdagents_service()
    
    valid_types = ["pcc", "specialist", "moderator", "recruiter", "reviewer", "integrator"]
    if agent_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"无效的智能体类型。可用类型: {valid_types}"
        )
    
    return service.get_agent_status(agent_type)


@router.get("/statistics", summary="获取服务统计信息")
async def get_statistics():
    """获取MDAgents服务使用统计信息"""
    service = get_mdagents_service()
    return service.get_statistics()


@router.post("/shutdown", summary="关闭服务")
async def shutdown_service(background_tasks: BackgroundTasks):
    """关闭MDAgents服务（仅用于维护）"""
    service = get_mdagents_service()
    
    async def shutdown():
        await service.shutdown()
    
    background_tasks.add_task(shutdown)
    
    return {"message": "MDAgents服务正在关闭..."}
