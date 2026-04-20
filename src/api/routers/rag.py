"""  
RAG服务路由  
提供知识检索和增强生成接口  
"""  

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path
import time

from schemas.rag import (  
    RetrieveRequest, RetrieveResponse,   
    RagQuestionRequest, RagQuestionResponse,  
    KnowledgeBaseRequest, KnowledgeBaseResponse,  
    DocumentUploadRequest  
)  
from services.rag_service import get_rag_service, RAGService  

router = APIRouter()  

@router.post("/retrieve", response_model=RetrieveResponse)  
async def retrieve_documents(  
    request: RetrieveRequest,  
    rag_service: RAGService = Depends(get_rag_service)  
):  
    """  
    检索相关文档  
    
    Args:  
        request: 检索请求  
        rag_service: RAG服务  
    
    Returns:  
        检索结果  
    """  
    start_time = time.time()  
    
    try:  
        # 执行检索
        documents = rag_service.retrieve(
            kb_name=request.kb_name,
            query=request.query,
            top_k=request.top_k,
            filter=request.filter
        )  
        
        # 处理结果
        
        process_time = time.time() - start_time
        
        return RetrieveResponse(
            query=request.query,
            documents=documents,
            kb_name=request.kb_name,
            process_time=process_time,
            total_results=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    


@router.post("/ask", response_model=RagQuestionResponse)
async def ask_rag_question(
    request: RagQuestionRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    基于知识库回答问题
    
    Args:
        request: RAG问题请求
        rag_service: RAG服务
    
    Returns:
        回答结果
    """
    start_time = time.time()
    
    try:
        # 生成回答
        response = rag_service.generate_response(
            kb_name=request.kb_name,
            query=request.question,
            model_name=request.model_name,
            top_k=request.top_k
        )
        
        # 处理结果
        process_time = time.time() - start_time
        
        return RagQuestionResponse(
            question=request.question,
            answer=response.get("answer", ""),
            contexts=response.get("contexts", []),
            sources=response.get("sources", []),
            kb_name=request.kb_name,
            model=request.model_name or "default",
            process_time=process_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kb", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    获取可用知识库列表
    
    Args:
        rag_service: RAG服务
    
    Returns:
        知识库列表
    """
    try:
        kbs = rag_service.get_available_knowledge_bases()
        return [KnowledgeBaseResponse(**kb) for kb in kbs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kb", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    创建新知识库
    
    Args:
        request: 知识库创建请求
        rag_service: RAG服务
    
    Returns:
        创建的知识库信息
    """
    try:
        success = rag_service.create_knowledge_base(
            kb_name=request.name,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=f"创建知识库失败: {request.name}")
        
        # 获取创建的知识库信息
        kbs = rag_service.get_available_knowledge_bases()
        for kb in kbs:
            if kb.get("name") == request.name:
                return KnowledgeBaseResponse(**kb)
        
        # 如果没找到，返回基本信息
        return KnowledgeBaseResponse(
            name=request.name,
            description=request.description,
            document_count=0,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/kb/{kb_name}", response_model=Dict[str, Any])
async def delete_knowledge_base(
    kb_name: str = Path(..., description="知识库名称"),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    删除知识库
    
    Args:
        kb_name: 知识库名称
        rag_service: RAG服务
    
    Returns:
        操作结果
    """
    try:
        success = rag_service.delete_knowledge_base(kb_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"删除知识库失败: {kb_name}")
        
        return {"success": True, "message": f"知识库已删除: {kb_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
    

@router.post("/kb/{kb_name}/documents", response_model=Dict[str, Any])
async def add_documents(
    request: DocumentUploadRequest,
    kb_name: str = Path(..., description="知识库名称"),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    向知识库添加文档
    
    Args:
        request: 文档上传请求
        kb_name: 知识库名称
        rag_service: RAG服务
    
    Returns:
        操作结果
    """
    try:
        success = rag_service.add_documents(kb_name, request.documents)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"添加文档失败: {kb_name}")
        
        return {
            "success": True, 
            "message": f"已添加 {len(request.documents)} 个文档到知识库: {kb_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))