import os
import sys
import json
from typing import List, Dict, Any, Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from models.api_model import ApiModel
from config.llm_config import LLMConfig



class MedEmbeddingDataGenerator:
    """ 医疗嵌入数据生成器"""

    def __init__(self, llm_config: LLMConfig, save_path: str, data_num: int = 1000):
        """
        初始化嵌入数据生成器
        
        Args:
            data_num: 生成的嵌入数据数量
        """
        self.data_num = data_num
        self.llm_config = llm_config
        self.api_model = ApiModel(llm_config)
        self.save_path = save_path