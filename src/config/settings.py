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
        # 读取 YAML 原始内容并替换环境变量占位符
        with open(config_path, 'r') as f:
            yaml_content = f.read()

        # 替换 ${VAR} 或 ${VAR:default} 形式的占位符
        import re
        def replace_var(match):
            var_expr = match.group(1)
            # 支持 ${VAR:default} 语法
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                return os.environ.get(var_name.strip(), default.strip() if default.strip() else '')
            else:
                return os.environ.get(var_expr.strip(), '')

        yaml_content = re.sub(r'\$\{([^}]+)\}', replace_var, yaml_content)

        self._config = OmegaConf.create(yaml_content)

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
