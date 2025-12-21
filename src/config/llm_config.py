import os, sys
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    model_provider: str = Field(...,description="The name of the model provider.")

    model_name: str = Field(...,description="The name of the model to use.")


    api_key: str = Field(..., description="The API key to use.")

    base_url: str = Field(..., description="The base URL to use.")

    max_tokens:int = Field(default=2048,description="The maximum number of tokens to generate.")

    max_input_tokens:int = Field(default=2048,description="The maximum number of tokens to generate.")

    temperature:float = Field(default=0.7,description="The temperature to use for sampling.")

    top_p:float = Field(default=0.9,description="The top-p value to use for sampling.")

    top_k:int = Field(default=50,description="The top-k value to use for sampling.")

    stream:bool = Field(default=False,description="Whether to use streaming mode.")

    timeout:int = Field(default=60,description="The timeout to use for requests.")


    def __post_init__(self):
        """在初始化后检查模型提供程序是否支持指定模型"""
        if self.model_provider == "qwen":
            self.model_name = os.getenv("QWEN_MODEL_ID")
            self.api_key = os.getenv("QWEN_API_KEY")
            self.base_url = os.getenv("QWEN_ENDPOINT")

        elif self.model_provider == "openai":
            self.model_name = os.getenv("GPT5_MODEL_ID")
            self.api_key = os.getenv("GPT5_API_KEY")
            self.base_url = os.getenv("GPT5_ENDPOINT")

        elif self.model_provider == "zhipuai" or self.model_provider == "zhipu":
            self.model_name = os.getenv("ZHIPU_MODEL_ID")
            self.api_key = os.getenv("ZHIPU_API_KEY")
            self.base_url = os.getenv("ZHIPU_ENDPOINT")

        else:
            raise ValueError(f"模型提供商 {self.model_provider} 不支持")


