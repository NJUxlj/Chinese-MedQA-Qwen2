

import os
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import uvicorn

from models.qwen_model import QwenModel
from models.api_model import ApiModel
from inference.fastllm_inference import FastLLMInference
from inference.vllm_inference import VLLMInference
from knowledge_base.embedding_manager import EmbeddingManager
from knowledge_base.retrieval.knn_retriever import KNNRetriever
from rag.rag_pipeline import RAGPipeline
from rag.response_generator import ResponseGenerator
from agent.medical_agent import MedicalAgent
from config.model_config import ModelConfig
from utils.logger import setup_logger

# 创建logger
logger = setup_logger(__name__)

# 加载配置
model_config = ModelConfig()

# 创建FastAPI应用
app = FastAPI(
    title="Chinese-MedQA-Qwen2 API",
    description="基于Qwen2+Agent+RAG的医疗问答系统API",
    version="1.0.0",
)

# 定义请求和响应模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    use_rag: bool = Field(True, description="是否使用RAG增强")
    use_agent: bool = Field(True, description="是否使用Agent")
    top_k: Optional[int] = Field(5, description="检索的文档数量")
    

class RagResponse(BaseModel):
    query: str = Field(..., description="原始查询")
    response: str = Field(..., description="生成的回答")
    context: Optional[str] = Field(None, description="RAG上下文")
    source_documents: Optional[List[Dict[str, Any]]] = Field(None, description="源文档")
    

# 全局模型和pipeline实例
model = None
rag_pipeline = None
response_generator = None
medical_agent = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化模型和服务"""
    global model, rag_pipeline, response_generator, medical_agent
    
    logger.info("正在初始化服务...")
    
    try:
        # 根据配置决定模型类型
        if model_config.use_api:
            logger.info("使用API模型")
            model = ApiModel(
                api_key=model_config.api_key,
                api_base=model_config.api_base,
                model_name=model_config.model_name
            )
        elif model_config.use_fastllm:
            logger.info("使用FastLLM加速推理")
            model = FastLLMInference(
                model_path=model_config.model_path,
                device=model_config.device
            )
        elif model_config.use_vllm:
            logger.info("使用VLLM加速推理")
            model = VLLMInference(
                model_path=model_config.model_path,
                tensor_parallel_size=model_config.tensor_parallel_size,
                gpu_memory_utilization=model_config.gpu_memory_utilization
            )
        else:
            logger.info("使用本地Qwen模型")
            model = QwenModel(
                model_path=model_config.model_path,
                device=model_config.device
            )
        
        # 初始化RAG pipeline
        logger.info("初始化RAG pipeline")
        rag_pipeline = RAGPipeline(
            retriever_type=model_config.retriever_type,
            embedding_model_name=model_config.embedding_model_name,
            index_path=model_config.index_path,
            top_k=model_config.top_k
        )
        
        # 初始化响应生成器
        response_generator = ResponseGenerator(
            model=model,
            max_new_tokens=model_config.max_new_tokens,
            temperature=model_config.temperature,
            top_p=model_config.top_p
        )
        
        # 初始化医疗Agent
        medical_agent = MedicalAgent(
            model=model,
            rag_pipeline=rag_pipeline,
            verbose=model_config.verbose
        )
        
        logger.info("服务初始化完成")
    
    except Exception as e:
        logger.error(f"初始化服务失败: {e}")
        raise RuntimeError(f"初始化服务失败: {e}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    if model is None or rag_pipeline is None:
        raise HTTPException(status_code=503, detail="服务未完全初始化")
    return {"status": "healthy", "service": "Chinese-MedQA-Qwen2"}


@app.post("/query", response_model=RagResponse)
async def query(request: QueryRequest):
    """查询接口"""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="查询不能为空")
        
        if request.use_agent:
            # 使用Agent进行处理
            if request.use_rag:
                # 使用RAG增强的Agent
                result = medical_agent.process_with_rag(request.query)
            else:
                # 普通Agent处理
                result = medical_agent.run(request.query)
            
            response = {
                "query": request.query,
                "response": result["response"],
                "context": result.get("rag_context"),
                "source_documents": None
            }
        else:
            # 不使用Agent，只使用RAG
            if request.use_rag:
                # 构建RAG提示
                rag_prompt = rag_pipeline.build_rag_prompt(
                    request.query, 
                    top_k=request.top_k
                )
                
                # 生成响应
                result = response_generator.generate(rag_prompt)
                
                response = {
                    "query": request.query,
                    "response": result["response"],
                    "context": rag_prompt["context"],
                    "source_documents": result.get("source_documents")
                }
            else:
                # 直接使用模型生成
                prompt = f"请回答以下医疗问题：\n\n问题：{request.query}\n\n回答："
                result = model.generate(
                    prompt,
                    max_new_tokens=model_config.max_new_tokens,
                    temperature=model_config.temperature,
                    top_p=model_config.top_p
                )
                
                response = {
                    "query": request.query,
                    "response": result,
                    "context": None,
                    "source_documents": None
                }
        
        return response
    
    except Exception as e:
        logger.error(f"处理查询时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")


if __name__ == "__main__":
    # 运行FastAPI应用
    uvicorn.run(
        "main:app", 
        host=model_config.host, 
        port=model_config.port,
        reload=model_config.debug
    )

