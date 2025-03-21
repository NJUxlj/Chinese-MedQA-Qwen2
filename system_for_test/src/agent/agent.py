
import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import yaml

from langchain.schema import Document

from src.llm import get_llm
from src.retrieval.pubmed import PubMedRetriever
from src.retrieval.knowledge_base import LocalKnowledgeBase
from src.retrieval.vector_store import MedicalVectorStore
from src.agent.reflection import ReflectionEngine

class MedicalAgent:
    """
    医疗AI Agent核心类，协调各组件实现完整功能
    """
    
    def __init__(self, config_path: str):
        """
        初始化医疗AI Agent。
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化组件
        self.llm = get_llm(self.config)
        self.pubmed_retriever = PubMedRetriever(self.config['pubmed'])
        self.knowledge_base = LocalKnowledgeBase(self.config['local_knowledge'])
        self.vector_store = MedicalVectorStore(self.config['vector_store'])
        self.reflection_engine = ReflectionEngine(self.llm, self.config['agent'])
        
        # 定义工具列表
        self.tools = self._define_tools()
        
        print("医疗AI Agent初始化完成")
        
    def _define_tools(self) -> List[Dict]:
        """
        定义Agent可用的工具。
        
        Returns:
            工具定义列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_pubmed",
                    "description": "在PubMed上搜索医学文献",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "PubMed搜索查询"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "最大返回结果数"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_local_knowledge",
                    "description": "搜索本地医疗知识库",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "知识库搜索查询"
                            },
                            "k": {
                                "type": "integer",
                                "description": "返回的结果数量"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_article_details",
                    "description": "获取特定PubMed文章的详细信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pmid": {
                                "type": "string",
                                "description": "PubMed ID"
                            }
                        },
                        "required": ["pmid"]
                    }
                }
            }
        ]
        
    def search_pubmed(self, query: str, max_results: Optional[int] = None) -> Dict:
        """
        执行PubMed搜索工具。
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            搜索结果字典
        """
        results = self.pubmed_retriever.search(query, max_results)
        return {
            "results": results,
            "total_count": len(results),
            "message": f"在PubMed上找到 {len(results)} 个相关结果"
        }
        
    def search_local_knowledge(self, query: str, k: int = 5) -> Dict:
        """
        搜索本地知识库工具。
        
        Args:
            query: 搜索查询
            k: 返回结果数
            
        Returns:
            搜索结果字典
        """
        results = self.vector_store.search(query, k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "filename": doc.metadata.get("filename", "未知")
            })
            
        return {
            "results": formatted_results,
            "total_count": len(results),
            "message": f"在本地知识库中找到 {len(results)} 个相关结果"
        }
        
    def fetch_article_details(self, pmid: str) -> Dict:
        """
        获取特定PubMed文章详情工具。
        
        Args:
            pmid: PubMed ID
            
        Returns:
            文章详情字典
        """
        article = self.pubmed_retriever.fetch_article_by_id(pmid)
        full_text = self.pubmed_retriever.fetch_full_text(pmid)
        
        if article:
            article["full_text_available"] = full_text is not None
            if full_text:
                article["full_text"] = full_text
                
            return {
                "article": article,
                "success": True,
                "message": f"成功获取文章 {pmid} 的详细信息"
            }
        else:
            return {
                "article": None,
                "success": False,
                "message": f"未找到ID为 {pmid} 的文章"
            }
            
    def index_local_documents(self):
        """
        处理并索引本地知识库中的文档。
        """
        print("开始处理本地知识库文档...")
        documents = self.knowledge_base.process_documents()
        
        if documents:
            print(f"正在为 {len(documents)} 个文档创建向量索引...")
            self.vector_store.add_documents(documents)
            print("本地知识库索引完成")
        else:
            print("没有文档可索引")
            
    def _parse_tool_calls(self, tool_calls) -> List[Dict]:
        """
        解析LLM返回的工具调用。
        
        Args:
            tool_calls: LLM返回的工具调用
            
        Returns:
            处理后的工具调用结果列表
        """
        results = []
        
        if not tool_calls:
            return results
            
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # 根据函数名调用相应方法
                if function_name == "search_pubmed":
                    result = self.search_pubmed(
                        query=function_args.get("query"),
                        max_results=function_args.get("max_results")
                    )
                elif function_name == "search_local_knowledge":
                    result = self.search_local_knowledge(
                        query=function_args.get("query"),
                        k=function_args.get("k", 5)
                    )
                elif function_name == "fetch_article_details":
                    result = self.fetch_article_details(
                        pmid=function_args.get("pmid")
                    )
                else:
                    result = {"error": f"未知工具: {function_name}"}
                    
                results.append({
                    "tool_call_id": tool_call.id,
                    "function_name": function_name,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.id if hasattr(tool_call, "id") else "unknown",
                    "function_name": tool_call.function.name if hasattr(tool_call, "function") else "unknown",
                    "result": {"error": f"工具调用失败: {str(e)}"}
                })
                
        return results
        
    def process_query(self, query: str, use_tools: bool = True, conversation_history: Optional[List] = None) -> Dict:
        """
        处理用户查询，根据需要使用工具和反思。
        
        Args:
            query: 用户查询
            use_tools: 是否使用工具
            conversation_history: 对话历史（可选）
            
        Returns:
            包含响应和处理细节的字典
        """
        start_time = time.time()
        
        # 构建系统提示
        system_prompt = """你是一位专业的医疗AI助手，专注于提供准确、可靠的医学信息。
你可以：
1. 搜索PubMed获取最新的医学研究
2. 查询本地医疗知识库
3. 获取特定医学文章的详情

请遵循以下原则：
- 始终以医学准确性为首要任务
- 清晰表述医学共识与争议
- 适当引用可靠来源支持你的回答
- 承认知识的局限性
- 避免医疗建议，强调咨询专业医生的重要性
- 使用简洁、清晰的语言解释复杂的医学概念

回答问题时，请首先考虑是否需要查询更多信息，并使用适当的工具获取信息。"""
        
        retrieved_contexts = []
        tool_call_results = []
        
        # 如果需要使用工具
        if use_tools:
            # 对查询使用工具进行增强
            tool_response = self.llm.generate_with_tools(
                prompt=query,
                tools=self.tools,
                system_prompt=system_prompt,
                temperature=self.config['agent'].get('temperature', 0.1)
            )
            
            content = tool_response.get("content", "")
            tool_calls = tool_response.get("tool_calls")
            
            # 如果有工具调用，执行它们
            if tool_calls:
                tool_call_results = self._parse_tool_calls(tool_calls)
                
                # 准备工具返回的上下文
                for result in tool_call_results:
                    function_name = result.get("function_name")
                    result_data = result.get("result", {})
                    
                    if function_name == "search_pubmed":
                        articles = result_data.get("results", [])
                        for article in articles[:3]:  # 限制使用的文章数量
                            context = f"标题: {article.get('title')}\n"
                            context += f"作者: {article.get('authors')}\n"
                            context += f"摘要: {article.get('abstract')}\n"
                            context += f"发表于: {article.get('journal')} ({article.get('publication_date')})\n"
                            context += f"PMID: {article.get('pmid')}\n"
                            retrieved_contexts.append(context)
                            
                    elif function_name == "search_local_knowledge":
                        docs = result_data.get("results", [])
                        for doc in docs:
                            context = f"来源: {doc.get('filename')}\n"
                            context += f"内容: {doc.get('content')}\n"
                            retrieved_contexts.append(context)
                            
                    elif function_name == "fetch_article_details":
                        article = result_data.get("article")
                        if article:
                            context = f"标题: {article.get('title')}\n"
                            context += f"作者: {article.get('authors')}\n"
                            context += f"摘要: {article.get('abstract')}\n"
                            
                            if article.get("full_text_available") and "full_text" in article:
                                # 截取部分全文以避免过长
                                full_text = article.get("full_text", "")
                                if len(full_text) > 2000:
                                    full_text = full_text[:2000] + "...(全文已截断)"
                                context += f"全文: {full_text}\n"
                                
                            retrieved_contexts.append(context)
                
                # 在工具调用后，生成最终回答
                query_with_context = f"{query}\n\n这是我找到的相关信息:\n\n" + "\n\n".join(retrieved_contexts)
                initial_response = self.llm.generate(
                    prompt=query_with_context,
                    system_prompt=system_prompt,
                    temperature=self.config['agent'].get('temperature', 0.1)
                )
            else:
                # 如果没有工具调用，直接使用模型回答
                initial_response = content
        else:
            # 不使用工具时直接生成回答
            initial_response = self.llm.generate(
                prompt=query,
                system_prompt=system_prompt,
                temperature=self.config['agent'].get('temperature', 0.1)
            )
            
        # 使用反思引擎改进回答
        final_response = self.reflection_engine.reflect_on_response(
            query=query,
            initial_response=initial_response,
            retrieved_context=retrieved_contexts
        )
        
        processing_time = time.time() - start_time
        
        # 返回完整结果
        return {
            "query": query,
            "response": final_response,
            "initial_response": initial_response,
            "tool_call_results": tool_call_results,
            "retrieved_contexts": retrieved_contexts,
            "processing_time": processing_time
        }
