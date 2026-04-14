from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import spacy
from config.settings import settings
from typing import List, Dict, Any, Optional


class NERService:
    """命名实体识别服务"""
    def __init__(self, config=None):
        """初始化命名实体识别服务"""
        self.config = config if config is not None else settings.ner
        self.model_provider = self.config.model_provider

        self.ner_model = None
        self.transformers_pipeline = None
        self._load_ner_model()

    def _load_ner_model(self) -> None:
        """加载命名实体识别模型"""
        if self.model_provider == "spacy":
            self._load_spacy_model()
        elif self.model_provider == "transformers":
            self._load_transformers_model()
        else:
            raise ValueError(f"Invalid model provider: {self.model_provider}")

    def _load_spacy_model(self) -> None:
        """加载 spaCy 模型"""
        try:
            self.ner_model = spacy.load(self.config.model_name)
        except OSError:
            print(f"spaCy model {self.config.model_name} not found. Please install it first.")
            print(f"Install with: python -m spacy download {self.config.model_name}")
            raise

    def _load_transformers_model(self) -> None:
        """加载 HuggingFace Transformers 模型"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

            self.transformers_pipeline = pipeline(
                "ner",
                model=self.config.model_name,
                aggregation_strategy=self.config.aggregation_strategy,
                device=self.config.device,
                token= self.config.use_auth_token
            )
        except Exception as e:
            print(f"Failed to load Transformers model {self.config.model_name}: {e}")
            raise

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取命名实体"""
        if self.model_provider == "spacy":
            return self._extract_entities_spacy(text)
        elif self.model_provider == "transformers":
            return self._extract_entities_transformers(text)
        else:
            raise ValueError(f"Invalid model provider: {self.model_provider}")

    def _extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """使用 spaCy 提取实体"""
        doc = self.ner_model(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "source": "spacy"
            })
        return entities

    def _extract_entities_transformers(self, text: str) -> List[Dict[str, Any]]:
        """使用 HuggingFace Transformers 提取实体"""
        if self.transformers_pipeline is None:
            raise RuntimeError("Transformers pipeline not loaded")

        results = self.transformers_pipeline(text)

        entities = []
        for result in results:
            entities.append({
                "text": result["word"],
                "start": result["start"],
                "end": result["end"],
                "label": result["entity_group"],
                "score": result["score"],
                "source": "transformers"
            })
        return entities

    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """专门用于医学文本的实体提取"""
        all_entities = self.extract_entities(text)

        medical_labels = {
            "DISEASE", "SYMPTOM", "DRUG", "MEDICAL_EQUIPMENT",
            "BODY_PART", "MEDICAL_PROCEDURE", "CHEMICAL",
            "PERSONDOC", "ORG", "GPE"
        }

        medical_entities = []
        for entity in all_entities:
            if entity.get("label") in medical_labels:
                medical_entities.append(entity)

        return medical_entities

    def batch_extract_entities(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """批量提取实体"""
        if self.model_provider == "transformers":
            batch_size = self.config.batch_size
            all_results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    results = self.extract_entities(text)
                    all_results.append(results)

            return all_results
        else:
            return [self.extract_entities(text) for text in texts]
