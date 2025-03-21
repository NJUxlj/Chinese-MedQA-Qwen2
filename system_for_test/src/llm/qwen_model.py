
import os
from typing import Dict, List, Optional, Union, Any

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class QwenLLM:
    """Qwen2大语言模型接口，支持本地模型和API调用"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qwen2 LLM。
        
        Args:
            config: 包含模型配置的字典
        """
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen2-7B-Instruct")
        self.api_key = config.get("api_key", os.environ.get("QWEN_API_KEY"))
        self.use_api = self.api_key is not None
        
        if self.use_api:
            # 使用API
            self.client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=self.api_key
            )
        else:
            # 本地模型
            print(f"正在加载本地Qwen2模型: {self.model_name}")
            self._load_local_model()
    
    def _load_local_model(self):
        """加载本地Qwen2模型"""
        # 配置量化参数以节省内存
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """
        生成文本响应。
        
        Args:
            prompt: 用户输入的提示
            system_prompt: 系统提示（可选）
            temperature: 生成温度，越低越确定性
            max_tokens: 最大生成令牌数
            stream: 是否流式输出
            
        Returns:
            生成的文本响应
        """
        if self.use_api:
            # 使用API调用
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name.split("/")[-1],  # 提取模型名称
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # 流式输出处理
                collected_chunks = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        collected_chunks.append(chunk.choices[0].delta.content)
                        # 如果需要实时显示，可以在这里添加打印
                return "".join(collected_chunks)
            else:
                return response.choices[0].message.content
        else:
            # 本地模型调用
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt or ""}, 
                 {"role": "user", "content": prompt}],
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response
            
    def generate_with_tools(
        self, 
        prompt: str, 
        tools: List[Dict], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Dict:
        """
        使用工具生成响应（支持函数调用）。
        
        Args:
            prompt: 用户输入的提示
            tools: 工具定义列表
            system_prompt: 系统提示（可选）
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            
        Returns:
            包含响应和可能工具调用的字典
        """
        if not self.use_api:
            raise NotImplementedError("本地模型暂不支持工具调用，请使用API模式")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name.split("/")[-1],
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "content": response.choices[0].message.content,
            "tool_calls": None
        }
        
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            result["tool_calls"] = response.choices[0].message.tool_calls
            
        return result
