import os
import sys
import json
from typing import List, Dict



class MedQaDataGenerator:
    """ 医疗问答数据生成器"""

    def __init__(self, data_num: int = 1000):
        """
        初始化问答数据生成器
        
        Args:
            data_num: 生成的问答数据数量
        """
        self.data_num = data_num



    def generate_doctor(self):
        pass


    def generate_patient(self):
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