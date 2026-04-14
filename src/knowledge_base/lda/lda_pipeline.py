'''
对一个文档库进行 LDA 主题建模

使用主流框架 BERTopic 进行主题建模，支持中文文本处理
'''
import os
import re
import logging
import pickle
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__)).parent.parent.parent)

from config.settings import settings
from utils.logger import setup_logger

# 导入本地PDF解析器（可选）
PdfParser = None
try:
    from knowledge_base.pdf.pdf_parser import PdfParser
except ImportError:
    try:
        # 如果相对导入失败，使用绝对路径导入
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_parser_path = os.path.join(current_dir, "..", "pdf", "pdf_parser.py")
        
        if os.path.exists(pdf_parser_path):
            sys.path.insert(0, os.path.dirname(pdf_parser_path))
            
            # 动态导入PdfParser
            import importlib.util
            spec = importlib.util.spec_from_file_location("pdf_parser", pdf_parser_path)
            pdf_parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pdf_parser_module)
            PdfParser = pdf_parser_module.PdfParser
        else:
            print("未找到PDF解析器，PDF功能将被禁用")
    except Exception as e:
        print(f"无法加载PDF解析器: {e}，PDF功能将被禁用")

warnings.filterwarnings("ignore")


# 设置中文字体（用于可视化）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LDAPipeline:
    """LDA 主题建模管道，基于 BERTopic 实现
    
    支持中文文本的主题建模，提供PDF文档处理和结果可视化功能
    """

    def __init__(self,
                 config=None):
        """
        初始化主题建模管道

        Args:
            embedding_model: 用于生成文档嵌入的模型路径或名称
            language: 处理语言，支持 'chinese' 和 'english'
            offline_mode: 是否使用离线模式（不使用在线模型）
        """
        self.logger = setup_logger(self.__class__.__name__)
        if config is None:
            # Create a simple config object with default values
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                'embedding_model_name': settings.embedding.model_name if hasattr(settings, 'embedding') else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'language': 'chinese',
                'offline_mode': False
            })
        self.config = config
        self.embedding_model_name = config.embedding_model_name if hasattr(config, 'embedding_model_name') else settings.embedding.model_name
        self.language = config.language if hasattr(config, 'language') else 'chinese'
        self.pdf_parser = PdfParser() if PdfParser else None
        self.embedding_model = None
        self.topic_model = None
        self.offline_mode = config.offline_mode if hasattr(config, 'offline_mode') else False
        
        # 中文停用词列表
        self.chinese_stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '能', '下', '过', '他', '来', '对', '时候', '后', '如果',
            '就是', '因为', '什么', '可以', '这个', '中', '么', '出', '比', '还', '把',
            '从', '给', '被', '让', '向', '各', '其', '当', '与', '关于', '等', '但是',
            '然后', '所以', '只是', '已经', '如果', '虽然', '由于', '因此', '总之'
        }
        
        # 初始化嵌入模型
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """初始化文档嵌入模型"""
        try:
            self.logger.info(f"正在加载嵌入模型: {self.embedding_model_name}")
            
            # 如果是本地路径，尝试从本地加载
            if Path(self.embedding_model_name).exists():
                self.logger.info("检测到本地模型路径，尝试从本地加载...")
                
                # 尝试从本地路径加载SentenceTransformer模型
                try:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                except Exception as local_error:
                    self.logger.warning(f"从本地路径加载失败: {local_error}")
                    # 尝试使用本地模型名称
                    model_name = Path(self.embedding_model_name).name
                    self.logger.info(f"尝试使用模型名称: {model_name}")
                    self.embedding_model = SentenceTransformer(model_name, cache_folder=str(Path(self.embedding_model_name).parent))
            else:
                # 在线下载模型
                self.logger.info("使用在线模型下载...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            self.logger.info("嵌入模型加载成功")
        except Exception as e:
            self.logger.error(f"加载嵌入模型失败: {e}")
            # 如果本地模型加载失败，尝试使用一个默认的中文模型
            self.logger.info("尝试使用默认的中文模型...")
            try:
                self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self.logger.info("默认模型加载成功")
            except Exception as fallback_error:
                self.logger.error(f"默认模型加载也失败: {fallback_error}")
                raise

    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符，保留中文、英文和数字
        if self.language == "chinese":
            # 中文文本：保留中文字符、英文和数字
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        else:
            # 英文文本：保留字母、数字和空格
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 转换为小写（英文）
        if self.language == "english":
            text = text.lower()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _get_vectorizer(self, max_features: int = 10000) -> CountVectorizer:
        """
        获取文本向量化器
        
        Args:
            max_features: 最大特征数
            
        Returns:
            配置好的向量化器
        """
        if self.language == "chinese":
            # 中文文本处理
            stop_words = list(self.chinese_stopwords)
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words=stop_words,
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                min_df=2,  # 至少在2个文档中出现
                max_df=0.95  # 最多在95%的文档中出现
            )
        else:
            # 英文文本处理
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        return vectorizer

    def modeling_pdfs(self, 
                     pdf_dir: str, 
                     num_topics: Optional[int] = None,
                     min_topic_size: int = 10,
                     top_n_words: int = 10,
                     max_features: int = 10000,
                     save_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        对 PDF 文档库进行主题建模

        Args:
            pdf_dir: PDF 文档目录
            num_topics: 主题数量，如果为None则自动确定
            min_topic_size: 每个主题的最小文档数
            top_n_words: 每个主题的top N个关键词
            max_features: 向量化最大特征数
            save_model_path: 保存模型的路径

        Returns:
            包含主题建模结果的字典
        """
        self.logger.info(f"开始对PDF目录进行主题建模: {pdf_dir}")
        
        if not self.pdf_parser:
            self.logger.error("PDF解析器不可用，请确保正确安装相关依赖")
            return {"error": "PDF解析器不可用，请确保正确安装相关依赖"}
        
        try:
            # 批量解析PDF文件
            documents = self.pdf_parser.batch_parse_pdfs(pdf_dir)
            
            if not documents:
                self.logger.warning("没有找到可处理的PDF文档")
                return {"error": "没有找到可处理的PDF文档"}
            
            # 提取文本内容
            texts = []
            for doc in documents:
                text = self._preprocess_text(doc.page_content)
                if text:  # 只保留非空文本
                    texts.append(text)
            
            if not texts:
                self.logger.warning("没有可用的文本内容进行建模")
                return {"error": "没有可用的文本内容进行建模"}
            
            self.logger.info(f"成功提取 {len(texts)} 个文档的文本内容")
            
            # 调用文档主题建模方法
            return self.modeling_documents(
                documents=texts,
                num_topics=num_topics,
                min_topic_size=min_topic_size,
                top_n_words=top_n_words,
                max_features=max_features,
                save_model_path=save_model_path,
                source_type="pdf"
            )
                
        except Exception as e:
            self.logger.error(f"PDF主题建模失败: {e}")
            return {"error": f"PDF主题建模失败: {str(e)}"}

    def modeling_documents(self, 
                          documents: List[str],
                          num_topics: Optional[int] = None,
                          min_topic_size: int = 2,
                          top_n_words: int = 10,
                          max_features: int = 10000,
                          save_model_path: Optional[str] = None,
                          source_type: str = "text") -> Dict[str, Any]:
        """
        对文档列表进行主题建模

        Args:
            documents: 文档列表，每个元素为一个字符串
            num_topics: 主题数量，如果为None则自动确定
            min_topic_size: 每个主题的最小文档数
            top_n_words: 每个主题的top N个关键词
            max_features: 向量化最大特征数
            save_model_path: 保存模型的路径
            source_type: 文档来源类型，用于日志记录

        Returns:
            包含主题建模结果的字典
        """
        self.logger.info(f"开始对{len(documents)}个{source_type}文档进行主题建模")
        
        try:
            # 预处理文档
            processed_docs = []
            for doc in documents:
                processed_text = self._preprocess_text(doc)
                if processed_text:  # 只保留非空文档
                    processed_docs.append(processed_text)
            
            if not processed_docs:
                self.logger.warning("预处理后没有可用文档")
                return {"error": "预处理后没有可用文档"}
            
            self.logger.info(f"预处理后剩余 {len(processed_docs)} 个有效文档")
            
            # 准备BERTopic参数，根据文档数量动态调整
            n_docs = len(processed_docs)
            
            # 对于小数据集，使用更简单的配置
            if n_docs <= 10:
                self.logger.info(f"小数据集模式: {n_docs} 个文档")
                # 对于非常小的数据集，完全禁用UMAP，直接使用嵌入
                umap_model = None
                n_components = 2  # 最小维度
            else:
                self.logger.info(f"标准模式: {n_docs} 个文档")
                n_neighbors = min(15, max(2, n_docs - 1))
                n_components = min(5, n_docs - 1)
                umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
            
            self.logger.info(f"文档数量: {n_docs}, UMAP模型: {umap_model is not None}")
            
            # HDBSCAN参数调整
            adjusted_min_cluster_size = min(min_topic_size, max(2, n_docs // 2))  # 不超过文档数的一半
            adjusted_min_samples = min(1, adjusted_min_cluster_size - 1)  # 确保min_samples < min_cluster_size
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=adjusted_min_cluster_size,
                min_samples=adjusted_min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            vectorizer_model = self._get_vectorizer(max_features)
            
            # 创建BERTopic模型
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                top_n_words=top_n_words,
                calculate_probabilities=True,
                verbose=True
            )
            
            # 拟合模型，对于小数据集使用备用方法
            if n_docs <= 5:
                self.logger.info("使用简单LDA方法处理小数据集...")
                result = self._simple_topic_modeling(processed_docs, num_topics, top_n_words)
                return result
            else:
                self.logger.info("使用BERTopic模型...")
                topics, probabilities = self.topic_model.fit_transform(processed_docs)
            
            # 调整主题数量
            if num_topics is not None:
                self.logger.info(f"将主题数量调整为 {num_topics}")
                topics = self.topic_model.reduce_outliers(processed_docs, topics)
                self.topic_model = self.topic_model.reduce_topics(processed_docs, topics, nr_topics=num_topics)
                topics, probabilities = self.topic_model.transform(processed_docs)
            
            # 获取主题信息
            topic_info = self.topic_model.get_topic_info()
            
            # 准备结果
            result = {
                "num_documents": len(processed_docs),
                "num_topics": len(topic_info[topic_info.Topic != -1]),  # 排除异常主题
                "topics": topics,
                "probabilities": probabilities.tolist() if len(probabilities) > 0 else [],
                "topic_info": topic_info.to_dict('records'),
                "model_fitted": True
            }
            
            # 获取每个主题的关键词
            topic_words = {}
            for topic_id in topic_info.Topic.unique():  # 从 topic_info DataFrame 中提取所有唯一的主题ID（包括正常主题和异常主题）
                if topic_id != -1:  # 排除异常主题
                    words = self.topic_model.get_topic(topic_id)
                    if words:
                        topic_words[topic_id] = [{"word": word, "score": score} for word, score in words[:top_n_words]]
            
            result["topic_words"] = topic_words
            
            # 保存模型
            if save_model_path:
                self.save_model(save_model_path)
                result["model_saved_path"] = save_model_path
            
            self.logger.info(f"主题建模完成，发现 {result['num_topics']} 个主题")
            
            return result
            
        except Exception as e:
            self.logger.error(f"文档主题建模失败: {e}")
            return {"error": f"文档主题建模失败: {str(e)}"}
    
    def _simple_topic_modeling(self, 
                              documents: List[str], 
                              num_topics: Optional[int] = None,
                              top_n_words: int = 10) -> Dict[str, Any]:
        """
        使用简单的LDA方法处理小数据集
        
        Args:
            documents: 文档列表
            num_topics: 主题数量
            top_n_words: 每个主题的关键词数量
            
        Returns:
            主题建模结果
        """
        self.logger.info("开始简单LDA主题建模...")
        
        try:
            n_docs = len(documents)
            
            # 确定主题数量
            if num_topics is None:
                num_topics = min(3, n_docs)  # 最多3个主题
            
            num_topics = min(num_topics, n_docs)  # 确保不超过文档数
            
            # 为小数据集创建特殊的向量化器
            if self.language == "chinese":
                # 中文文本处理，使用更宽松的参数
                stop_words = list(self.chinese_stopwords)
                vectorizer = TfidfVectorizer(
                    max_features=min(500, n_docs * 50),  # 减少特征数
                    stop_words=stop_words,
                    ngram_range=(1, 1),  # 只使用1-gram
                    min_df=1,  # 最小文档频率
                    max_df=0.95  # 最大文档频率
                )
            else:
                # 英文文本处理
                vectorizer = TfidfVectorizer(
                    max_features=min(500, n_docs * 50),
                    stop_words='english',
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=0.95
                )
            
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # 检查是否有特征
            if doc_term_matrix.shape[1] == 0:
                self.logger.warning("向量化后没有特征，尝试更宽松的参数...")
                # 使用更宽松的参数重新向量化
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    min_df=1,
                    max_df=1.0,  # 允许所有词
                    token_pattern=r'(?u)\b\w+\b' if self.language == "chinese" else None
                )
                doc_term_matrix = vectorizer.fit_transform(documents)
                
                if doc_term_matrix.shape[1] == 0:
                    return {"error": "无法提取有效的特征词，请检查文本内容"}
            
            # 使用LDA主题建模
            lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10,  # 减少迭代次数以适应小数据集
                learning_method='online'
            )
            
            # 拟合模型
            lda_model.fit(doc_term_matrix)
            
            # 获取主题-词汇分布
            feature_names = vectorizer.get_feature_names_out()
            
            # 准备结果
            topic_words = {}
            topic_info = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                # 获取top词汇
                top_word_indices = topic.argsort()[-top_n_words:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_word_indices]
                
                topic_words[topic_idx] = [
                    {"word": word, "score": float(score)} 
                    for word, score in top_words
                ]
                
                topic_info.append({
                    "Topic": topic_idx,
                    "Name": f"主题 {topic_idx}",
                    "Count": 0,  # 简单LDA不计算文档数
                    "Percentage": 0.0,
                    "Representation": top_words[:5]
                })
            
            # 计算文档-主题分布
            doc_topic_dist = lda_model.transform(doc_term_matrix)
            topics = doc_topic_dist.argmax(axis=1)
            
            # 更新主题信息中的文档统计
            topic_counts = np.bincount(topics, minlength=num_topics)
            for i, count in enumerate(topic_counts):
                topic_info[i]["Count"] = int(count)
                topic_info[i]["Percentage"] = float(count / n_docs * 100)
            
            result = {
                "num_documents": n_docs,
                "num_topics": num_topics,
                "topics": topics.tolist(),
                "probabilities": doc_topic_dist.tolist(),
                "topic_info": topic_info,
                "topic_words": topic_words,
                "model_fitted": True,
                "method": "simple_lda"
            }
            
            self.logger.info(f"简单LDA主题建模完成，发现 {num_topics} 个主题")
            return result
            
        except Exception as e:
            self.logger.error(f"简单LDA主题建模失败: {e}")
            return {"error": f"简单LDA主题建模失败: {str(e)}"}

    def visualize_topics(self, 
                        save_path: Optional[str] = None,
                        width: int = 1200,
                        height: int = 800) -> Optional[str]:
        """
        可视化主题模型结果
        
        Args:
            save_path: 保存图像的路径
            width: 图像宽度
            height: 图像高度
            
        Returns:
            保存的图像路径（如果指定了save_path）
        """
        if not self.topic_model:
            self.logger.error("模型尚未拟合，请先运行主题建模")
            return None
        
        try:
            self.logger.info("正在生成主题可视化...")
            
            # 生成主题可视化
            fig = self.topic_model.visualize_topics(width=width, height=height)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"主题可视化已保存到: {save_path}")
                return save_path
            else:
                # 显示图像
                fig.show()
                return None
                
        except Exception as e:
            self.logger.error(f"生成主题可视化失败: {e}")
            return None

    def visualize_topic_distribution(self, 
                                   topics: List[int],
                                   save_path: Optional[str] = None) -> Optional[str]:
        """
        可视化主题分布
        
        Args:
            topics: 主题标签列表
            save_path: 保存图像的路径
            
        Returns:
            保存的图像路径
        """
        if not self.topic_model:
            self.logger.error("模型尚未拟合")
            return None
        
        try:
            self.logger.info("正在生成主题分布可视化...")
            
            fig = self.topic_model.visualize_topic_distribution(topics)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"主题分布可视化已保存到: {save_path}")
                return save_path
            else:
                fig.show()
                return None
                
        except Exception as e:
            self.logger.error(f"生成主题分布可视化失败: {e}")
            return None

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取特定主题的关键词
        
        Args:
            topic_id: 主题ID
            top_n: 返回的关键词数量
            
        Returns:
            关键词及其权重列表
        """
        if not self.topic_model:
            self.logger.error("模型尚未拟合")
            return []
        
        try:
            return self.topic_model.get_topic(topic_id)[:top_n]
        except Exception as e:
            self.logger.error(f"获取主题 {topic_id} 关键词失败: {e}")
            return []

    def save_model(self, save_path: str):
        """
        保存训练好的模型
        
        Args:
            save_path: 保存路径
        """
        if not self.topic_model:
            self.logger.error("模型尚未拟合，无法保存")
            return
        
        try:
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存BERTopic模型
            self.topic_model.save(save_path)
            self.logger.info(f"模型已保存到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")

    def load_model(self, load_path: str):
        """
        加载预训练的模型
        
        Args:
            load_path: 模型文件路径
        """
        try:
            self.topic_model = BERTopic.load(load_path)
            self.logger.info(f"模型已从 {load_path} 加载")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")

    def export_results(self, 
                      result: Dict[str, Any], 
                      export_path: str,
                      format: str = "json"):
        """
        导出主题建模结果
        
        Args:
            result: 主题建模结果
            export_path: 导出路径
            format: 导出格式，支持 'json' 和 'csv'
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                # 导出为JSON格式
                import json
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            elif format.lower() == "csv":
                # 导出为CSV格式（主要导出主题信息）
                if "topic_info" in result:
                    topic_df = pd.DataFrame(result["topic_info"])
                    topic_df.to_csv(export_path, index=False, encoding='utf-8')
                    
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"结果已导出到: {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出结果失败: {e}")

    def analyze_topic_trends(self, 
                           documents: List[str], 
                           timestamps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析主题趋势（需要时间戳信息）
        
        Args:
            documents: 文档列表
            timestamps: 时间戳列表（可选）
            
        Returns:
            主题趋势分析结果
        """
        if not self.topic_model:
            self.logger.error("模型尚未拟合")
            return {"error": "模型尚未拟合"}
        
        try:
            # 这里可以实现主题随时间变化的分析
            # 暂时返回基础信息
            topic_info = self.topic_model.get_topic_info()
            
            result = {
                "total_topics": len(topic_info[topic_info.Topic != -1]),
                "topic_sizes": topic_info[topic_info.Topic != -1]['Count'].to_dict(),
                "topic_names": topic_info[topic_info.Topic != -1]['Name'].to_dict()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析主题趋势失败: {e}")
            return {"error": f"分析主题趋势失败: {str(e)}"}





def run():
    pass



if __name__ == '__main__':
    run()



