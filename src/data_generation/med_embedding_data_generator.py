import os
import sys
import json
from typing import List, Dict, Any, Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from models.api_model import ApiModel
from config.llm_config import LLMConfig



class MedEmbeddingDataGenerator:
    """ 医疗嵌入数据生成器"""

    def __init__(self, llm_config: LLMConfig, save_path: str, data_num: int = 1000):
        """
        初始化嵌入数据生成器
        
        Args:
            data_num: 生成的嵌入数据数量
        """
        self.data_num = data_num
        self.llm_config = llm_config
        self.api_model = ApiModel(llm_config)
        self.save_path = save_path


    def generate_positive_sentence_pairs(self, collection_names:List[str]):
        '''
        生成正例对， 将来我们会利用对比学习，对嵌入模型进行微调

        怎么做：

            # 做法 1
            我们从 milvus 向量库中挑选若干个 collection， 对于一个 colleciton 中的任意两个 text chunk,
                如果他们的 metadata 中的 page 相同， 我们就将他们作为正例对

            # 做法 2
            我们从 milvus 向量库中挑选若干个 collection， 对于每一个 collection，我们对其中的所有 text-chunk 进行 LDA 主题建模，得到每个 text-chunk 的主题分布。
                我们将同一个主题下的 text-chunk 作为正例对
        
        Returns:
            List[Dict]: 包含正例对的列表，每个正例对包含两个句子
                Dict = {
                    "sentence1": "这是一个句子",
                    "sentence2": "这是另一个句子"
                }
        '''
        pass




    def generate_negative_sentence_pairs(self):
        '''
        生成负例对， 将来我们会利用对比学习，对嵌入模型进行微调

        怎么做：

            # 做法 1
            我们从 milvus 向量库中挑选若干个 collection， 对于一个 colleciton 中的任意两个 text chunk,
                如果他们的 metadata 中的 page 不相同， 我们就将他们作为负例对

            # 做法 2
            我们从 milvus 向量库中挑选若干个 collection， 对于每一个 collection，我们对其中的所有 text-chunk 进行 LDA 主题建模，得到每个 text-chunk 的主题分布。
                我们将不同主题下的 text-chunk 作为负例对
        
        Returns:
            List[Dict]: 包含负例对的列表，每个负例对包含两个句子
                Dict = {
                    "sentence1": "这是一个句子",
                    "sentence2": "这是另一个句子"
                }
        '''

    def merge_and_save_sentence_pairs(self):
        pass