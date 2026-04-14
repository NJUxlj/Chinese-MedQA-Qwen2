from omegaconf import OmegaConf
from pathlib import Path
import os

class Settings:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = Path(__file__).parent / "config.yaml"
        self._config = OmegaConf.load(config_path)
        # 环境变量覆盖
        self._config = OmegaConf.merge(
            self._config,
            OmegaConf.from_dotlist([f"{k}={v}" for k, v in os.environ.items()])
        )

    @property
    def llm(self): return self._config.llm
    @property
    def milvus(self): return self._config.milvus
    @property
    def rag(self): return self._config.rag
    @property
    def agent(self): return self._config.agent
    @property
    def embedding(self): return self._config.embedding
    @property
    def inference(self): return self._config.inference

settings = Settings()
