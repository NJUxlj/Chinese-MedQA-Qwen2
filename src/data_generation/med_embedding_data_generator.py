import os
import sys
import json
import random
from typing import List, Dict, Any, Union, Literal, Tuple
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.api_model import ApiModel
from config.llm_config import LLMConfig
from knowledge_base.milvus.milvus_client import MilvusClient
from knowledge_base.lda.lda_pipeline import LDAPipeline
from config.lda_config import LDAConfig
from langchain_core.documents import Document


class MedEmbeddingDataGenerator:
    """ 医疗嵌入数据生成器
    
    用于生成对比学习训练数据，支持从 Milvus 向量库中提取文档，
    并通过页面分组或 LDA 主题建模生成正例对和负例对。
    """

    def __init__(self, llm_config: LLMConfig, save_path: str, data_num: int = 1000):
        """
        初始化嵌入数据生成器
        
        Args:
            llm_config: LLM 配置
            save_path: 数据保存路径
            data_num: 生成的嵌入数据数量
        """
        self.data_num = data_num
        self.llm_config = llm_config
        self.api_model = ApiModel(llm_config)
        self.save_path = save_path
        
        # 初始化 Milvus 客户端
        self.milvus_client = MilvusClient()
        
        # 初始化 LDA 配置和管道
        self.lda_config = LDAConfig(
            embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
            language="chinese",
            offline_mode=False
        )
        self.lda_pipeline = LDAPipeline(config=self.lda_config)


    def _connect_milvus(self) -> bool:
        """连接到 Milvus 服务器"""
        return self.milvus_client.connect()


    def _disconnect_milvus(self):
        """断开与 Milvus 的连接"""
        self.milvus_client.disconnect()


    def _load_collection_documents(self, collection_name: str) -> List[Document]:
        """
        加载指定 Collection 的所有文档
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            List[Document]: 文档列表
        """
        try:
            # 初始化向量存储
            if not self.milvus_client.init_vector_store(collection_name):
                return []
            
            # 获取 Collection 信息
            info = self.milvus_client.get_collection_info(collection_name)
            num_entities = info.get("num_entities", 0) if info else 0
            
            if num_entities == 0:
                return []
            
            # 搜索所有文档（使用大 k 值获取所有文档）
            all_docs = self.milvus_client.similarity_search(
                collection_name=collection_name,
                query="",
                k=min(num_entities, 10000)
            )
            
            return all_docs
            
        except Exception as e:
            print(f"加载 Collection 文档失败: {e}")
            return []


    def generate_positive_sentence_pairs(
        self, 
        collection_names: List[str],
        method: Literal["by_page", "by_lda"],
        positive_ratio: float = 1.0
    ) -> List[Dict[str, str]]:
        """
        生成正例对，用于对比学习训练
        
        Args:
            collection_names: Milvus 中的 Collection 名称列表
            method: 生成方法
                - "by_page": 基于页面分组，同一页面的不同 chunk 作为正例对
                - "by_lda": 基于 LDA 主题建模，同一主题下的 chunk 作为正例对
            positive_ratio: 正例对占总数据的比例
            
        Returns:
            List[Dict]: 包含正例对的列表
                [{"sentence1": "句子1", "sentence2": "句子2", "source": {"collection": "xxx", "page": "1"}}]
        """
        if not self._connect_milvus():
            print("无法连接到 Milvus 服务器")
            return []
        
        try:
            all_positive_pairs = []
            
            for collection_name in collection_names:
                if not self.milvus_client.collection_exists(collection_name):
                    print(f"Collection {collection_name} 不存在，跳过")
                    continue
                
                print(f"正在处理 Collection: {collection_name}")
                
                # 加载文档
                documents = self._load_collection_documents(collection_name)
                
                if not documents:
                    print(f"Collection {collection_name} 为空，跳过")
                    continue
                
                # 根据方法生成正例对
                if method == "by_page":
                    pairs = self._generate_pairs_by_page(documents, collection_name)
                else:  # by_lda
                    pairs = self._generate_pairs_by_lda(documents, collection_name)
                
                all_positive_pairs.extend(pairs)
            
            # 根据比例截取数据
            target_num = int(self.data_num * positive_ratio)
            if len(all_positive_pairs) > target_num:
                all_positive_pairs = random.sample(all_positive_pairs, target_num)
            
            print(f"生成了 {len(all_positive_pairs)} 个正例对")
            return all_positive_pairs
            
        finally:
            self._disconnect_milvus()


    def _generate_pairs_by_page(
        self, 
        documents: List[Document], 
        collection_name: str
    ) -> List[Dict[str, str]]:
        """
        基于页面分组生成正例对
        
        同一页面的不同 chunk 被认为是语义相似的，作为正例对
        
        Args:
            documents: 文档列表
            collection_name: Collection 名称
            
        Returns:
            正例对列表
        """
        # 按页面分组
        page_groups: Dict[str, List[Document]] = {}
        
        for doc in documents:
            # 从 metadata 中获取 page 信息
            metadata = doc.metadata
            page = metadata.get("page", metadata.get("source", "unknown"))
            
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        # 生成正例对（同一页面内的任意两个不同 chunk）
        positive_pairs = []
        
        for page, docs in page_groups.items():
            if len(docs) < 2:
                continue
            
            # 随机采样生成对
            num_pairs = min(len(docs) * (len(docs) - 1) // 2, 50)  # 限制每个页面的对数
            
            if num_pairs == 0:
                continue
                
            # 生成所有可能的组合
            all_combinations = []
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    all_combinations.append((i, j))
            
            # 随机选择
            if len(all_combinations) > num_pairs:
                selected = random.sample(all_combinations, num_pairs)
            else:
                selected = all_combinations
            
            for i, j in selected:
                pair = {
                    "sentence1": docs[i].page_content,
                    "sentence2": docs[j].page_content,
                    "source": {
                        "collection": collection_name,
                        "page": page,
                        "method": "by_page"
                    }
                }
                positive_pairs.append(pair)
        
        return positive_pairs


    def _generate_pairs_by_lda(
        self, 
        documents: List[Document], 
        collection_name: str
    ) -> List[Dict[str, str]]:
        """
        基于 LDA 主题建模生成正例对
        
        同一主题下的 chunk 被认为是语义相似的，作为正例对
        
        Args:
            documents: 文档列表
            collection_name: Collection 名称
            
        Returns:
            正例对列表
        """
        # 提取文本
        texts = [doc.page_content for doc in documents]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(texts) if text and len(text.strip()) > 10]
        
        if len(valid_indices) < 2:
            return []
        
        valid_texts = [texts[i] for i in valid_indices]
        valid_docs = [documents[i] for i in valid_indices]
        
        print(f"  有效文档数: {len(valid_texts)}")
        
        # LDA 主题建模
        try:
            result = self.lda_pipeline.modeling_documents(
                documents=valid_texts,
                num_topics=None,
                min_topic_size=2,
                top_n_words=10
            )
            
            if "error" in result:
                print(f"  LDA 建模失败: {result['error']}")
                return []
            
            topics = result.get("topics", [])
            topic_info = result.get("topic_info", [])
            
            print(f"  发现 {len([t for t in topics if t != -1])} 个主题")
            
            # 按主题分组
            topic_groups: Dict[int, List[int]] = {}
            for idx, topic_id in enumerate(topics):
                if topic_id == -1:
                    continue  # 跳过异常主题
                if topic_id not in topic_groups:
                    topic_groups[topic_id] = []
                topic_groups[topic_id].append(idx)
            
            # 生成正例对（同一主题下的任意两个不同 chunk）
            positive_pairs = []
            
            for topic_id, indices in topic_groups.items():
                if len(indices) < 2:
                    continue
                
                # 限制每个主题的对数
                max_pairs_per_topic = 50
                num_pairs = min(len(indices) * (len(indices) - 1) // 2, max_pairs_per_topic)
                
                if num_pairs == 0:
                    continue
                
                # 生成所有可能的组合
                all_combinations = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        all_combinations.append((indices[i], indices[j]))
                
                # 随机选择
                if len(all_combinations) > num_pairs:
                    selected = random.sample(all_combinations, num_pairs)
                else:
                    selected = all_combinations
                
                for i, j in selected:
                    pair = {
                        "sentence1": valid_texts[i],
                        "sentence2": valid_texts[j],
                        "source": {
                            "collection": collection_name,
                            "topic_id": topic_id,
                            "method": "by_lda"
                        }
                    }
                    positive_pairs.append(pair)
            
            return positive_pairs
            
        except Exception as e:
            print(f"  LDA 处理失败: {e}")
            return []


    def generate_negative_sentence_pairs(
        self, 
        collection_names: List[str],
        method: Literal["by_page", "by_lda"],
        negative_ratio: float = 1.0,
        hard_negative_ratio: float = 0.3
    ) -> List[Dict[str, str]]:
        """
        生成负例对，用于对比学习训练
        
        Args:
            collection_names: Milvus 中的 Collection 名称列表
            method: 生成方法
                - "by_page": 基于页面分组，不同页面的 chunk 作为负例对
                - "by_lda": 基于 LDA 主题建模，不同主题下的 chunk 作为负例对
            negative_ratio: 负例对占总数据的比例
            hard_negative_ratio: 困难负例的比例（语义相近但主题不同的对）
            
        Returns:
            List[Dict]: 包含负例对的列表
                [{"sentence1": "句子1", "sentence2": "句子2", "source": {...}, "type": "hard/soft"}]
        """
        if not self._connect_milvus():
            print("无法连接到 Milvus 服务器")
            return []
        
        try:
            all_negative_pairs = []
            
            for collection_name in collection_names:
                if not self.milvus_client.collection_exists(collection_name):
                    print(f"Collection {collection_name} 不存在，跳过")
                    continue
                
                print(f"正在处理 Collection: {collection_name}")
                
                # 加载文档
                documents = self._load_collection_documents(collection_name)
                
                if not documents:
                    print(f"Collection {collection_name} 为空，跳过")
                    continue
                
                # 根据方法生成负例对
                if method == "by_page":
                    pairs = self._generate_negative_pairs_by_page(documents, collection_name)
                else:  # by_lda
                    pairs = self._generate_negative_pairs_by_lda(documents, collection_name, hard_negative_ratio)
                
                all_negative_pairs.extend(pairs)
            
            # 根据比例截取数据
            target_num = int(self.data_num * negative_ratio)
            if len(all_negative_pairs) > target_num:
                all_negative_pairs = random.sample(all_negative_pairs, target_num)
            
            print(f"生成了 {len(all_negative_pairs)} 个负例对")
            return all_negative_pairs
            
        finally:
            self._disconnect_milvus()


    def _generate_negative_pairs_by_page(
        self, 
        documents: List[Document], 
        collection_name: str
    ) -> List[Dict[str, str]]:
        """
        基于页面分组生成负例对
        
        不同页面的 chunk 被认为是语义不相似的，作为负例对
        
        Args:
            documents: 文档列表
            collection_name: Collection 名称
            
        Returns:
            负例对列表
        """
        # 按页面分组
        page_groups: Dict[str, List[Document]] = {}
        
        for doc in documents:
            metadata = doc.metadata
            page = metadata.get("page", metadata.get("source", "unknown"))
            
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        pages = list(page_groups.keys())
        
        if len(pages) < 2:
            return []
        
        # 生成负例对（不同页面之间的 chunk）
        negative_pairs = []
        max_pairs_per_collection = 100  # 限制每个 collection 的对数
        
        pair_count = 0
        attempts = 0
        max_attempts = 1000
        
        while pair_count < max_pairs_per_collection and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的页面
            page1, page2 = random.sample(pages, 2)
            
            # 从每个页面随机选择一个 chunk
            doc1 = random.choice(page_groups[page1])
            doc2 = random.choice(page_groups[page2])
            
            # 避免添加重复对
            pair = {
                "sentence1": doc1.page_content,
                "sentence2": doc2.page_content,
                "source": {
                    "collection": collection_name,
                    "page1": page1,
                    "page2": page2,
                    "method": "by_page"
                },
                "type": "soft"  # 不同页面的通常是简单负例
            }
            
            negative_pairs.append(pair)
            pair_count += 1
        
        return negative_pairs


    def _generate_negative_pairs_by_lda(
        self, 
        documents: List[Document], 
        collection_name: str,
        hard_negative_ratio: float = 0.3
    ) -> List[Dict[str, str]]:
        """
        基于 LDA 主题建模生成负例对
        
        不同主题下的 chunk 作为负例对，包含困难负例和简单负例
        
        Args:
            documents: 文档列表
            collection_name: Collection 名称
            hard_negative_ratio: 困难负例的比例
            
        Returns:
            负例对列表
        """
        # 提取文本
        texts = [doc.page_content for doc in documents]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(texts) if text and len(text.strip()) > 10]
        
        if len(valid_indices) < 2:
            return []
        
        valid_texts = [texts[i] for i in valid_indices]
        valid_docs = [documents[i] for i in valid_indices]
        
        # LDA 主题建模
        try:
            result = self.lda_pipeline.modeling_documents(
                documents=valid_texts,
                num_topics=None,
                min_topic_size=2,
                top_n_words=10
            )
            
            if "error" in result:
                print(f"  LDA 建模失败: {result['error']}")
                return []
            
            topics = result.get("topics", [])
            
            # 按主题分组
            topic_groups: Dict[int, List[int]] = {}
            for idx, topic_id in enumerate(topics):
                if topic_id == -1:
                    continue
                if topic_id not in topic_groups:
                    topic_groups[topic_id] = []
                topic_groups[topic_id].append(idx)
            
            topic_ids = list(topic_groups.keys())
            
            if len(topic_ids) < 2:
                return []
            
            # 区分困难负例和简单负例
            # 困难负例：主题在语义上相近（通过主题词重叠度判断）
            # 简单负例：主题明显不同
            
            hard_negative_pairs = []
            soft_negative_pairs = []
            
            max_total_pairs = 100
            max_hard_pairs = int(max_total_pairs * hard_negative_ratio)
            max_soft_pairs = max_total_pairs - max_hard_pairs
            
            # 计算主题相似度（基于主题词重叠）
            topic_similarities = self._calculate_topic_similarities(result.get("topic_words", {}))
            
            pair_count = 0
            hard_count = 0
            soft_count = 0
            attempts = 0
            max_attempts = 2000
            
            while (hard_count < max_hard_pairs or soft_count < max_soft_pairs) and attempts < max_attempts:
                attempts += 1
                
                topic1, topic2 = random.sample(topic_ids, 2)
                similarity = topic_similarities.get((topic1, topic2), topic_similarities.get((topic2, topic1), 0))
                
                # 判断是困难负例还是简单负例
                is_hard = similarity > 0.3  # 主题词重叠度 > 30% 认为是困难负例
                
                if is_hard and hard_count >= max_hard_pairs:
                    continue
                if not is_hard and soft_count >= max_soft_pairs:
                    continue
                
                # 随机选择一个 chunk 从每个主题
                idx1 = random.choice(topic_groups[topic1])
                idx2 = random.choice(topic_groups[topic2])
                
                pair = {
                    "sentence1": valid_texts[idx1],
                    "sentence2": valid_texts[idx2],
                    "source": {
                        "collection": collection_name,
                        "topic1": topic1,
                        "topic2": topic2,
                        "method": "by_lda"
                    },
                    "type": "hard" if is_hard else "soft",
                    "topic_similarity": similarity
                }
                
                if is_hard:
                    hard_negative_pairs.append(pair)
                    hard_count += 1
                else:
                    soft_negative_pairs.append(pair)
                    soft_count += 1
            
            return hard_negative_pairs + soft_negative_pairs
            
        except Exception as e:
            print(f"  LDA 处理失败: {e}")
            return []


    def _calculate_topic_similarities(
        self, 
        topic_words: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[Tuple[int, int], float]:
        """
        计算主题之间的相似度（基于主题词的 Jaccard 相似度）
        
        Args:
            topic_words: 主题词汇字典 {topic_id: [{"word": "词", "score": 0.5}, ...]}
            
        Returns:
            主题对相似度字典
        """
        topic_ids = list(topic_words.keys())
        similarities = {}
        
        # 提取每个主题的词集合
        topic_word_sets = {}
        for topic_id in topic_ids:
            words = set(item["word"] for item in topic_words.get(topic_id, []))
            topic_word_sets[topic_id] = words
        
        # 计算每对主题的相似度
        for i in range(len(topic_ids)):
            for j in range(i + 1, len(topic_ids)):
                t1, t2 = topic_ids[i], topic_ids[j]
                words1 = topic_word_sets[t1]
                words2 = topic_word_sets[t2]
                
                if not words1 or not words2:
                    similarity = 0.0
                else:
                    # Jaccard 相似度
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities[(t1, t2)] = similarity
        
        return similarities


    def merge_and_save_sentence_pairs(
        self, 
        positive_pairs: List[Dict[str, str]], 
        negative_pairs: List[Dict[str, str]],
        output_format: str = "json",
        positive_negative_ratio: Tuple[int, int] = (1, 1)
    ) -> str:
        """
        合并正例对和负例对，并保存到文件
        
        Args:
            positive_pairs: 正例对列表
            negative_pairs: 负例对列表
            output_format: 输出格式 ("json", "csv", "jsonl")
            positive_negative_ratio: 正负例比例，如 (1, 1) 表示 1:1
            
        Returns:
            str: 保存的文件路径
        """
        # 平衡正负例数量
        pos_count = len(positive_pairs)
        neg_count = len(negative_pairs)
        
        if pos_count == 0 and neg_count == 0:
            print("没有数据可保存")
            return ""
        
        # 按比例平衡数据
        ratio = positive_negative_ratio[0] / positive_negative_ratio[1] if positive_negative_ratio[1] > 0 else 1
        
        balanced_positive = positive_pairs
        balanced_negative = negative_pairs
        
        if ratio != 1:
            if ratio > 1:
                # 正例多，减少正例
                target = min(len(negative_pairs) * ratio, len(positive_pairs))
                balanced_positive = random.sample(positive_pairs, int(target))
            else:
                # 负例多，减少负例
                target = min(len(positive_pairs) / ratio if ratio > 0 else len(negative_pairs), len(negative_pairs))
                balanced_negative = random.sample(negative_pairs, int(target))
        
        # 合并数据
        merged_data = []
        
        # 添加正例对（label=1）
        for pair in balanced_positive:
            merged_data.append({
                "sentence1": pair["sentence1"],
                "sentence2": pair["sentence2"],
                "label": 1,
                "source": pair.get("source", {}),
                "type": "positive"
            })
        
        # 添加负例对（label=0）
        for pair in balanced_negative:
            merged_data.append({
                "sentence1": pair["sentence1"],
                "sentence2": pair["sentence2"],
                "label": 0,
                "source": pair.get("source", {}),
                "type": pair.get("type", "negative")
            })
        
        # 随机打乱顺序
        random.shuffle(merged_data)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # 保存文件
        base_path = self.save_path
        if not base_path.endswith(f".{output_format}"):
            base_path = f"{base_path}.{output_format}"
        
        if output_format == "json":
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
                
        elif output_format == "jsonl":
            with open(base_path, "w", encoding="utf-8") as f:
                for item in merged_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
        elif output_format == "csv":
            import csv
            with open(base_path, "w", encoding="utf-8", newline="") as f:
                if merged_data:
                    writer = csv.DictWriter(f, fieldnames=["sentence1", "sentence2", "label", "type"])
                    writer.writeheader()
                    writer.writerows(merged_data)
        else:
            print(f"不支持的格式: {output_format}")
            return ""
        
        print(f"数据已保存到: {base_path}")
        print(f"  总样本数: {len(merged_data)}")
        print(f"  正例数: {len([d for d in merged_data if d['label'] == 1])}")
        print(f"  负例数: {len([d for d in merged_data if d['label'] == 0])}")
        
        return base_path


    def generate_training_data(
        self,
        collection_names: List[str],
        method: Literal["by_page", "by_lda"] = "by_lda",
        positive_negative_ratio: Tuple[int, int] = (1, 1),
        output_format: str = "jsonl"
    ) -> str:
        """
        一键生成训练数据
        
        Args:
            collection_names: Milvus 中的 Collection 名称列表
            method: 生成方法
            positive_negative_ratio: 正负例比例
            output_format: 输出格式
            
        Returns:
            str: 保存的文件路径
        """
        print("=" * 50)
        print("开始生成嵌入模型训练数据")
        print("=" * 50)
        
        # 生成正例对
        print("\n[1/3] 生成正例对...")
        positive_pairs = self.generate_positive_sentence_pairs(collection_names, method)
        
        # 生成负例对
        print("\n[2/3] 生成负例对...")
        negative_pairs = self.generate_negative_sentence_pairs(collection_names, method)
        
        # 合并并保存
        print("\n[3/3] 合并并保存数据...")
        output_path = self.merge_and_save_sentence_pairs(
            positive_pairs, 
            negative_pairs,
            output_format=output_format,
            positive_negative_ratio=positive_negative_ratio
        )
        
        print("\n" + "=" * 50)
        print("训练数据生成完成!")
        print("=" * 50)
        
        return output_path


def run():
    """
    运行示例
    """
    from config.llm_config import LLMConfig
    
    # 配置
    llm_config = LLMConfig(
        model_name="Qwen2-7B-Instruct",
        api_key="your-api-key",
        base_url="http://localhost:8000/v1"
    )
    
    save_path = "./data/embedding_training_data"
    data_num = 1000
    
    # 创建生成器
    generator = MedEmbeddingDataGenerator(
        llm_config=llm_config,
        save_path=save_path,
        data_num=data_num
    )
    
    # Collection 名称列表
    collection_names = [
        "medical_knowledge_base",
        "clinical_guidelines",
        "pharmaceutical_data"
    ]
    
    # 生成训练数据（使用 LDA 方法）
    output_path = generator.generate_training_data(
        collection_names=collection_names,
        method="by_lda",
        positive_negative_ratio=(1, 1),
        output_format="jsonl"
    )
    
    print(f"\n输出文件: {output_path}")


if __name__ == "__main__":
    run()