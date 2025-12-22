import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from omegaconf import OmegaConf


class BaseConfig(BaseModel):
    '''
    基础配置类
    '''

    



    @classmethod
    def from_yaml(cls, yaml_file: str):
        '''
        从 YAML 文件加载配置
        '''
        conf = OmegaConf.load(yaml_file)
        return cls(**conf)