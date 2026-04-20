"""
统一配置入口

整个项目的所有配置一律从 ``src/config/config.yaml`` 加载，
不再读取任何环境变量、不再使用 ``.env`` 与 ``python-dotenv``。

使用方式::

    from config.settings import settings
    print(settings.llm.api_key)
    print(settings.api_server.port)

任意新增/调整配置项时，只需修改 ``config.yaml`` 即可，
无需在代码中再做映射。
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class Settings:
    """单例配置对象，所有顶层节均通过属性访问。"""

    _instance = None
    _config: DictConfig = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}. 请确认 src/config/config.yaml 存在并已填好所有配置项。"
            )
        self._config = OmegaConf.load(config_path)

    def __getattr__(self, name: str):
        """将 settings.xxx 直接转发到 config.yaml 顶层节。"""
        if name.startswith("_"):
            raise AttributeError(name)
        config = object.__getattribute__(self, "_config")
        if config is None or name not in config:
            raise AttributeError(
                f"配置项 '{name}' 不存在于 config.yaml 中。请在 src/config/config.yaml 中补充。"
            )
        return config[name]

    def __contains__(self, name: str) -> bool:
        return self._config is not None and name in self._config

    def as_dict(self):
        """以普通 dict 形式导出当前全部配置（用于调试/日志）。"""
        return OmegaConf.to_container(self._config, resolve=True)


settings = Settings()
