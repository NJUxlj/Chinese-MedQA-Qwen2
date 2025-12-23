import os
import sys
import json
from typing import List, Dict, Any, Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from models.api_model import ApiModel
from config.llm_config import LLMConfig
from config.milvus_config import MilvusConfig
from knowledge_base.milvus.milvus_client import MilvusClient
from knowledge_base.lda.lda_pipeline import LDAPipeline

class MedQaDataGenerator:
    """ 医疗问答数据生成器"""

    def __init__(self, 
        milvus_config: MilvusConfig,
        llm_config: LLMConfig, 
        save_path: str, 
        data_num: int = 1000):
        """
        初始化问答数据生成器
        
        Args:
            data_num: 生成的问答数据数量
        """
        self.data_num = data_num
        self.llm_config = llm_config
        self.api_model = ApiModel(llm_config)
        self.save_path = save_path

        self.mbti_personality_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]

        self.milvus_client = MilvusClient(milvus_config)
        self.llm = ApiModel(llm_config)

        self.doctor_gen_prompt = None
        self.patient_gen_prompt = None



    def generate_doctor(self, subtopic: str):
        '''
        根据 MBTI 16 人格来生成医生的背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 使用 reranker 重排序， 再取 top-kk 个最相关的文档 (top-kk < top-k)
        3. 调用 LLM 生成医生的背景资料， 将文档段， mbti 人格类型 作为 context 封装进 prompt， 生成医生的背景资料

        '''
        context = None


    def generate_patient(self, subtopic: str):
        '''
        根据 MBTI 16 人格来生成患者的背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 使用 reranker 重排序， 再取 top-kk 个最相关的文档 (top-kk < top-k)
        3. 调用 LLM 生成患者的背景资料， 将文档段， mbti 人格类型 作为 context 封装进 prompt， 生成患者的背景资料

        '''
        context = None
    

    def load_documents_from_milvus(self, collection_name:str):
        pass


    def modeling_topics_using_milvus(self):
        pass

    def generate_qa_topics(self, num_topics: int = 5):
        """
        生成问答主题
        
        Args:
            num_topics: 生成的医疗主题数量
            
        Returns:
            List[Dict]: 包含主题ID和名称的列表
        """
        topics = []
        for i in range(num_topics):
            topics.append({
                "id": i,
                "name": f"topic_{i}"
            })
        return topics


    def generate_subtopics(self, num_subtopics: int = 3):
        """
        生成子主题
        
        Args:
            num_subtopics: 生成的子主题数量
            
        Returns:
            List[Dict]: 包含子主题ID和名称的列表
        """
        subtopics = []
        for i in range(num_subtopics):
            subtopics.append({
                "id": i,
                "name": f"subtopic_{i}"
            })
        return subtopics



    def retrive_subtopic_documents(self, subtopics: List[str]):
        """
        检索子主题文档
        
        Args:
            subtopics: 子主题列表
            
        Returns:
            Dict: 包含子主题ID和文档列表的字典
        """
        subtopic_docs = {}
        for subtopic in subtopics:
            subtopic_docs[subtopic["id"]] = []
        return subtopic_docs



    def generate_doc_pat_conversation_foreach_subtopic(self, subtopic_docs: Dict[int, List[str]]):
        """
        为每个子主题生成医生-患者对话
        
        Args:
            subtopic_docs: 包含子主题ID和文档列表的字典
            
        Returns:
            Dict: 包含子主题ID和对话列表的字典
        """
        doc_pat_conversations = {}
        for subtopic_id, docs in subtopic_docs.items():
            doc_pat_conversations[subtopic_id] = []
        return doc_pat_conversations




    def generate_qa_data(self):
        pass



    def save_generated_qa_data(self):
        pass