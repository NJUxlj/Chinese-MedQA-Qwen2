from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import spacy
from config.ner_config import NERConfig
from typing import List, Dict, Any



class NERService:
    """命名实体识别服务"""
    def __init__(self, config: NERConfig):
        """初始化命名实体识别服务"""
        self.config = config
        self.model_provider = self.config.model_provider  # spacy or transformers

        self.ner_model = None
        self._load_ner_model()

    def _load_ner_model(self) -> None:
        """加载命名实体识别模型"""
        if self.model_provider == "spacy":
            self.ner_model = spacy.load(self.config.model_name)
        elif self.model_provider == "transformers":
            pass
        else:
            raise ValueError(f"Invalid model provider: {self.model_provider}")
        

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取命名实体"""
        doc = self.ner_model(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        return entities