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
from knowledge_base.milvus.milvus_client import MilvusClient
from knowledge_base.lda.lda_pipeline import LDAPipeline
from knowledge_base.kg.kg_client import KGClient
from rag.rag_pipeline import RAGPipeline


from config.milvus_config import MilvusConfig
from config.kg_config import KGConfig
from config.lda_config import LDAConfig






class QACorrector:
    def __init__(self):
        pass



class EvidenceBasedVerifier:
    def __init__(self):
        pass




class MedQaDataGenerator:
    """ 医疗问答数据生成器
    
    
    流程:
    1. 对 Milvus 中的若干个 collection:
        - 对其中的每个 collection， 我们首先进行主题建模， 得到 num_topics 个主题 （父主题）
        - 对每个父主题下面的所有文档，我们再逐一进行主题建模，得到子主题 （sub-topic）
        - 对每个子主题，我们根据 MBTI 16 人格类型， 生成对应的医生和患者的背景资料 （每个子主题对应要生成 16 条样本）
        - 对于每个子主题 + 人格 的组合， 我们需要单独为其生成医患对话 [但是，第一轮必须是患者问]。
            - 生成的过程中使用到了 RAG 技术， 我们并不是一次生成所有对话，而是一轮一轮的生成【一轮对话就相当于 messages 中的某个 role 发出的 content】。首先，在生成每一轮之前，我们使用 retrive_subtopic_documents 从 当前的 collection 中选出 top-k 个最相似片段。
            - 我们一轮一轮进行生成， 每生成完一轮对话 (医生，或者患者)， 都要结合知识图谱中的子图搜索功能、milvus 中的医学教科书集合、模型自身的医学常识。进行循证验证 （evidence-based verifier, 基于 ApiModel）。【验证的时候，输入当前生成的轮次，以及之前生成的所有轮次】
            - 如果验证不通过， 历史生成的对话，当前的对话，反馈信息， 会被输入到一个 QACorrector 进行修复。修复完继续传给 evidence-based verifier 进行验证。
            - 如果验证通过， 则继续生成下一轮对话。
    2. 对剩余的 所有 collections 都执行上述步骤。
    3. 合并所有 collection 对应的样本。
    4. 将所有的样本格式都转换为 OpenAI 的 messages 格式 （role:..., content:...）
    
    
    """

    def __init__(self, 
        milvus_config: MilvusConfig,
        llm_config: LLMConfig, 
        save_path: str, 
        data_num: int = 1000,
        textbook_collection_name:str = None):
        """
        初始化问答数据生成器
        
        Args:
            data_num: 生成的问答数据数量
        """
        self.data_num = data_num
        self.llm_config = llm_config
        self.api_model = ApiModel(llm_config)
        self.save_path = save_path

        self.textbook_collection_name = textbook_collection_name

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
        生成医生的背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 使用 reranker 重排序， 再取 top-kk 个最相关的文档 (top-kk < top-k)
        3. 调用 LLM 生成医生的背景资料， 将文档段， mbti 人格类型 作为 context 封装进 prompt， 生成医生的背景资料

        '''
        context = None


    def generate_patient(self, subtopic: str):
        '''
        同时结合相关文档片段和 MBTI 16 人格来生成患者的背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 使用 reranker 重排序， 再取 top-kk 个最相关的文档 (top-kk < top-k)
        3. 调用 LLM 生成患者的背景资料， 将文档段， mbti 人格类型 作为 context 封装进 prompt， 生成患者的背景资料

        '''
        context = None
    

    def load_documents_from_milvus(self, collection_name:str):
        pass


    def modeling_topics_using_lda(self):
        pass

    def generate_qa_topics(self, num_topics: int = 5):
        """
        生成问答主题
        
        Args:
            num_topics: 生成的医疗主题数量
            
        Returns:
            List[Dict]: 包含主题ID和名称的列表
        """
        pass


    def generate_subtopics(self, num_subtopics: int = 3):
        """
        生成子主题
        
        Args:
            num_subtopics: 生成的子主题数量
            
        Returns:
            List[Dict]: 包含子主题ID和名称的列表
        """
        pass



    def retrive_subtopic_documents(self, subtopics: List[str]):
        """
        检索子主题文档
        
        Args:
            subtopics: 子主题列表
            
        Returns:
            Dict: 包含子主题ID和文档列表的字典
        """
        pass



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
        """
        生成QA数据的主流程
        
        Returns:
            List[Dict]: 包含问答数据的列表
        """
        pass



    def save_generated_qa_data(self):
        pass