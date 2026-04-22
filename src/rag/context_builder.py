# src/rag/context_builder.py
import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from providers.reranker_provider import RerankerProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ContextBuilder:
    """
    上下文构建器，负责从检索结果构建增强上下文
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        reranker_model_provider: str = "huggingface",
        reranker_model_path: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        reranker_base_url: Optional[str] = None,
        reranker_api_key: Optional[str] = None,
        max_context_length: int = 4000,
        format_template: Optional[str] = None
    ):
        """
        初始化上下文构建器

        Args:
            chunk_size: 上下文块大小
            chunk_overlap: 上下文块重叠大小
            reranker_model_provider: 重排序模型 provider
            reranker_model_path: 重排序模型本地路径
            reranker_model_name: 重排序模型名称
            reranker_base_url: vLLM API 地址（优先于 model_path）
            reranker_api_key: vLLM API 密钥
            max_context_length: 最大上下文长度
            format_template: 格式化模板
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = max_context_length

        # 设置格式化模板
        if format_template:
            self.format_template = format_template
        else:
            self.format_template = (
                "请根据以下信息回答问题。如果无法从提供的信息中找到答案，请基于可靠的医学知识回答，"
                "并注明这是基于一般医学知识的回答。\n\n相关信息：\n{context}\n\n问题：{query}\n\n回答："
            )

        # 初始化重排序模型
        self.reranker_service = None

        # 优先使用 vLLM API（base_url 配置优先）
        if reranker_base_url and reranker_model_name:
            try:
                # 构建临时 config 供 RerankerProvider 使用
                class VLLMConfig:
                    model_provider = "vllm"
                    model_name = reranker_model_name
                    base_url = reranker_base_url
                    api_key = reranker_api_key or ""
                    batch_size = 8
                    normalize_scores = True

                self.reranker_service = RerankerProvider(VLLMConfig())
                logger.info(f"已加载 vLLM API reranker: {reranker_base_url}")
            except Exception as e:
                logger.error(f"加载 vLLM API reranker 失败: {e}")

        # 回退到本地模型
        elif reranker_model_name and reranker_model_path:
            try:
                class LocalConfig:
                    model_provider = reranker_model_provider
                    model_name = reranker_model_name
                    model_path = reranker_model_path
                    device = "cpu"
                    batch_size = 8
                    normalize_scores = True

                self.reranker_service = RerankerProvider(LocalConfig())
                logger.info(f"已加载重排序模型: {reranker_model_name}")
            except Exception as e:
                logger.error(f"加载重排序模型失败: {e}")
    
    def split_document(self, text: str) -> List[str]:
        """
        将文档分割成小块
        
        Args:
            text: 文档文本
            
        Returns:
            文本块列表
        """
        paragraphs = re.split(r'\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 匹配“中文/英文句号、感叹号、问号、分号”作为句子结束符；末尾的 ? 表示该结束符可出现 0 次或 1 次，
                # 从而兼容没有标点结尾的剩余文本
                sentence_pattern = r'[^.。!！?？;；]*[.。!！?？;；]?'
                sentence_matches = re.findall(sentence_pattern, paragraph)
                sentences = [s for s in sentence_matches if s.strip()]
                
                if not sentences:
                    sentences = [paragraph]
                
                current_sentence = ""
                for sentence in sentences:
                    if len(current_sentence) + len(sentence) <= self.chunk_size:
                        current_sentence += sentence
                    else:
                        if current_sentence:
                            chunks.append(current_sentence)
                        current_sentence = sentence
                
                if current_sentence:
                    chunks.append(current_sentence)
            
            elif len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += "\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用重排序模型对文档进行重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重新排序后的文档列表
        """
        if not self.reranker_service:
            logger.warning("未初始化重排序模型，跳过重排序")
            return documents
        
        document_texts = [doc["text"] for doc in documents]

        # 将 document_texts 封装为 document 对象列表
        original_metadata = [doc.get("metadata", {}) for doc in documents]
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(document_texts, original_metadata)]
        
        # 重排序
        reranked_documents = self.reranker_service.rerank(query = query, documents = documents, top_k = len(documents))
        
        # 将 Document 对象转换回字典格式
        reranked_dicts = []
        for doc in reranked_documents:
            reranked_dicts.append({
                "id": doc.metadata.get("doc_id", "unknown"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": doc.metadata.get("rerank_score", 0.0)
            })
        
        return reranked_dicts
    
    def build_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        use_reranker: bool = False
    ) -> str:
        """
        从文档列表构建上下文
        
        Args:
            query: 查询文本
            documents: 文档列表
            use_reranker: 是否使用重排序
            
        Returns:
            构建的上下文
        """
        if not documents:
            logger.warning("没有文档用于构建上下文")
            return ""
        
        # 如果要使用重排序
        if use_reranker and self.reranker_service:
            documents = self.rerank_documents(query, documents)
        
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # 提取文档文本
            text = doc.get("text", "") if isinstance(doc, dict) else doc.page_content
            
            # 如果文档过长，分割成小块
            if len(text) > self.chunk_size:
                chunks = self.split_document(text)
                
                # 添加每个块作为单独的部分
                for j, chunk in enumerate(chunks):
                    if current_length + len(chunk) + 100 <= self.max_context_length:  # 100是额外标记的空间
                        score_info = ""
                        if "score" in doc:
                            score_info = f" (相关度: {doc['score']:.2f})"
                        elif "rerank_score" in doc:
                            score_info = f" (相关度: {doc['rerank_score']:.2f})"
                        
                        source_info = ""
                        if "metadata" in doc and doc["metadata"]:
                            if "source" in doc["metadata"]:
                                source_info = f" [来源: {doc['metadata']['source']}]"
                        
                        # 构建标记
                        part_header = f"[文档 {i+1}-{j+1}]{score_info}{source_info}:\n"
                        part = f"{part_header}{chunk}\n\n"
                        
                        context_parts.append(part)
                        current_length += len(part)
                    else:
                        # 上下文已满
                        break
            else:
                # 直接添加整个文档
                if current_length + len(text) + 100 <= self.max_context_length:
                    score_info = ""
                    if "score" in doc:
                        score_info = f" (相关度: {doc['score']:.2f})"
                    elif "rerank_score" in doc:
                        score_info = f" (相关度: {doc['rerank_score']:.2f})"
                    
                    source_info = ""
                    if "metadata" in doc and doc["metadata"]:
                        if "source" in doc["metadata"]:
                            source_info = f" [来源: {doc['metadata']['source']}]"
                    
                    # 构建标记
                    part_header = f"[文档 {i+1}]{score_info}{source_info}:\n"
                    part = f"{part_header}{text}\n\n"
                    
                    context_parts.append(part)
                    current_length += len(part)
                else:
                    # 上下文已满
                    break
        
        # 拼接上下文
        context = "".join(context_parts)
        
        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length]
            logger.warning(f"上下文过长，已截断至 {self.max_context_length} 字符")
        
        return context.strip()
    
    def format_prompt(self, query: str, context: str) -> str:
        """
        格式化提示
        
        Args:
            query: 查询文本
            context: 上下文
            
        Returns:
            格式化后的提示
        """
        return self.format_template.format(query=query, context=context)
    
    def extract_key_information(self, context: str, query: str) -> str:
        """
        从上下文中提取与查询最相关的关键信息
        
        Args:
            context: 上下文
            query: 查询
            
        Returns:
            提取的关键信息
        """
        if not context:
            return ""
        
        # 分割上下文为不同的文档块
        doc_pattern = r'\[文档 \d+(?:-\d+)?\](?:\s*\(相关度:[0-9.]+\))?(?:\s*\[来源: [^\]]+\])?:\n(.*?)(?=\n\n\[文档 \d+|\Z)'
        doc_matches = re.finditer(doc_pattern, context, re.DOTALL)
        
        # 提取文档内容
        doc_contents = []
        for match in doc_matches:
            content = match.group(1).strip()
            if content:
                doc_contents.append(content)
        
        # 如果无法分割，直接返回上下文
        if not doc_contents:
            return context
        
        # 如果有重排序模型，使用它来选择最相关的段落
        if self.reranker_service:
            # 准备输入
            documents = [Document(page_content = content) for content in doc_contents]
            
            # 计算相关性分数
            sorted_contents = self.reranker_service.rerank(query = query, documents = documents, top_k = len(documents))

            sorted_contents = [doc.page_content for doc in sorted_contents]

            
            # 选择最相关的内容，控制总长度
            key_info = ""
            current_length = 0
            
            for content in sorted_contents:
                if current_length + len(content) + 10 <= self.max_context_length // 2:
                    key_info += content + "\n\n"
                    current_length += len(content) + 2
                else:
                    break
            
            return key_info.strip()
        else:
            # 如果没有重排序模型，简单连接前几个文档内容
            combined_content = "\n\n".join(doc_contents[:3])
            
            # 如果内容过长，截断
            if len(combined_content) > self.max_context_length // 2:
                return combined_content[:self.max_context_length // 2] + "..."
            
            return combined_content