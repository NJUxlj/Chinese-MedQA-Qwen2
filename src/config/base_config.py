import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from omegaconf import OmegaConf



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 解决 macOS上OpenMP库冲突的问题


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