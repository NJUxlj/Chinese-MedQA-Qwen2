from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import spacy
from config.kg_config import KGConfig
from typing import List, Dict, Any



class KGClient:
    """知识图谱客户端"""
    def __init__(self, config: KGConfig):
        """初始化知识图谱客户端"""
        self.config = config


    

    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识图谱中的实体"""
        pass



    def search_relations(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识图谱中的关系"""
        pass



    def search_subgraphs(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识图谱中的子图"""
        pass


    def entity_linking(self, query: str) -> List[Dict[str, Any]]:
        """实体链接"""
        pass