# src/rag/query_processor.py

import re
from typing import Dict, List, Optional, Union, Any, Tuple
import jieba
import jieba.analyse
from utils.text_utils import TextNormalizer, TextSegmenter
from utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryProcessor:
    """
    查询处理器，负责查询的清洗、扩展和优化
    """
    
    def __init__(
        self,
        use_query_expansion: bool = False,
        use_stopwords: bool = True,
        stopwords_path: Optional[str] = None,
        medical_dict_path: Optional[str] = None,
        max_query_length: int = 200,
        remove_punctuation: bool = False
    ):
        """
        初始化查询处理器
        
        Args:
            use_query_expansion: 是否使用查询扩展
            use_stopwords: 是否使用停用词
            stopwords_path: 停用词文件路径
            medical_dict_path: 医学词典路径
            max_query_length: 最大查询长度
            remove_punctuation: 是否移除标点符号
        """
        self.use_query_expansion = use_query_expansion
        self.use_stopwords = use_stopwords
        self.max_query_length = max_query_length
        self.remove_punctuation = remove_punctuation
        
        # 文本标准化器
        self.normalizer = TextNormalizer()
        
        # 文本分词器
        self.segmenter = TextSegmenter(user_dict_path=medical_dict_path)
        
        # 加载停用词
        self.stopwords = set()
        if use_stopwords and stopwords_path:
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = set([line.strip() for line in f])
                logger.info(f"已加载 {len(self.stopwords)} 个停用词")
            except Exception as e:
                logger.warning(f"加载停用词失败: {e}")
        
        # 加载医学词典
        if medical_dict_path:
            try:
                jieba.load_userdict(medical_dict_path)
                logger.info(f"已加载医学词典: {medical_dict_path}")
            except Exception as e:
                logger.warning(f"加载医学词典失败: {e}")
    
    def clean_query(self, query: str) -> str:
        """
        清理查询文本
        
        Args:
            query: 原始查询文本
            
        Returns:
            清理后的查询文本
        """
        # 基本清理
        query = self.normalizer.normalize_text(
            query,
            lowercase=False,  # 保留大小写（医学术语可能区分大小写）
            remove_html=True,
            remove_url=True,
            remove_emoji=True,
            normalize_unicode=True,
            normalize_spaces=True,
            normalize_punct=not self.remove_punctuation,  # 如果不移除标点则标准化标点
            to_half_width=True
        )
        
        # 如果需要移除标点符号
        if self.remove_punctuation:
            # 保留问号，因为在医疗查询中问号可能很重要
            query = re.sub(r'[^\w\s\u4e00-\u9fff?？]', '', query)
        
        # 截断过长的查询
        if len(query) > self.max_query_length:
            logger.warning(f"查询过长，已截断: {len(query)} -> {self.max_query_length}")
            query = query[:self.max_query_length]
        
        return query.strip()
    
    def remove_stopwords(self, query_tokens: List[str]) -> List[str]:
        """
        移除停用词
        
        Args:
            query_tokens: 查询分词结果
            
        Returns:
            移除停用词后的分词列表
        """
        if not self.use_stopwords or not self.stopwords:
            return query_tokens
        
        return [token for token in query_tokens if token not in self.stopwords]
    
    def expand_query(self, query: str, tokens: List[str]) -> str:
        """
        查询扩展，添加相关医学术语
        
        Args:
            query: 原始查询
            tokens: 查询分词结果
            
        Returns:
            扩展后的查询
        """
        if not self.use_query_expansion:
            return query
        
        # 使用TextRank提取关键词
        keywords = jieba.analyse.textrank(query, topK=3, withWeight=True)
        
        # 如果没有提取到关键词，返回原查询
        if not keywords:
            return query
        
        # 添加权重较高的关键词到查询中
        expanded_terms = []
        for word, weight in keywords:
            if word not in tokens and weight > 0.2:  # 只添加权重高且不在原查询中的词
                expanded_terms.append(word)
        
        # 如果有扩展词，添加到查询末尾
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.debug(f"查询扩展: {query} -> {expanded_query}")
            return expanded_query
        
        return query
    
    def extract_medical_concepts(self, query: str) -> List[str]:
        """
        从查询中提取医学概念
        
        Args:
            query: 查询文本
            
        Returns:
            提取的医学概念列表
        """
        # 使用医学词典进行分词，获取可能的医学概念
        words = list(jieba.cut(query))
        
        # 使用TextRank算法提取关键词，这些更可能是医学概念
        keywords = jieba.analyse.textrank(query, topK=5, withWeight=True)
        
        # 提取权重较高的关键词
        concepts = [word for word, weight in keywords if weight > 0.1]
        
        return concepts
    
    def process_query(self, query: str) -> str:
        """
        完整的查询处理流程
        
        Args:
            query: 原始查询文本
            
        Returns:
            处理后的查询文本
        """
        # 清理查询
        cleaned_query = self.clean_query(query)
        
        # 分词
        tokens = self.segmenter.segment(cleaned_query)
        
        # 移除停用词
        if self.use_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # 重新组合查询
        processed_query = " ".join(tokens)
        
        # 查询扩展
        if self.use_query_expansion:
            processed_query = self.expand_query(processed_query, tokens)
        
        logger.debug(f"查询处理: {query} -> {processed_query}")
        return processed_query
    
    def get_query_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """
        获取查询中的关键词
        
        Args:
            query: 查询文本
            top_k: 返回的关键词数量
            
        Returns:
            关键词列表
        """
        # 清理查询
        cleaned_query = self.clean_query(query)
        
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(cleaned_query, topK=top_k)
        
        return keywords
    
    def rewrite_query(self, query: str) -> str:
        """
        重写查询，优化检索效果
        
        Args:
            query: 原始查询
            
        Returns:
            重写后的查询
        """
        # 清理查询
        cleaned_query = self.clean_query(query)
        
        # 提取医学概念
        concepts = self.extract_medical_concepts(cleaned_query)
        
        # 如果没有提取到医学概念，返回原查询
        if not concepts:
            return cleaned_query
        
        # 将医学概念与原查询组合
        rewritten_query = f"{cleaned_query} {' '.join(concepts)}"
        
        logger.debug(f"查询重写: {query} -> {rewritten_query}")
        return rewritten_query