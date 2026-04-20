"""
LLM Provider - 统一的大语言模型调用接口

支持以下供应商:
- openai: OpenAI 兼容 API (model_name, base_url, api_key)
- aliyun: 阿里云百炼 API (model_name, base_url, api_key)
- vllm: vLLM 部署的模型 API (model_name, base_url, api_key)
- local: 本地 transformers 模型 (model_path)
"""

import sys
import logging
from typing import Dict, List, Optional, Union, Any, Literal
from pathlib import Path

import requests
import torch

# Add project root to path for utils import
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    统一 LLM Provider，支持多种供应商和本地模型加载。

    供应商类型 (provider):
        - openai: OpenAI 兼容 API
        - aliyun: 阿里云百炼 API
        - vllm: vLLM 部署的模型 API
        - local: 本地 transformers 模型
    """

    def __init__(
        self,
        provider: Literal["openai", "aliyun", "vllm", "local"] = "openai",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 2048,
        timeout: int = 60,
        stream: bool = False,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        max_context_length: int = 8192,
        **kwargs,
    ):
        """
        初始化 LLM Provider。

        Args:
            provider: 供应商类型，支持 openai/aliyun/vllm/local
            model_name: 模型名称（API 模式必填）
            base_url: API 基础 URL（API 模式必填）
            api_key: API 密钥（API 模式必填）
            model_path: 本地模型路径（local 模式必填）
            temperature: 生成温度
            top_p: 核采样参数
            top_k: top-k 采样参数
            max_tokens: 最大生成长度
            timeout: 请求超时时间（秒）
            stream: 是否使用流式输出
            device: 运行设备（cuda/cpu/mps），仅 local 模式有效
            load_in_8bit: 是否量化到 8bit，仅 local 模式有效
            load_in_4bit: 是否量化到 4bit，仅 local 模式有效
            trust_remote_code: 是否信任远程代码，仅 local 模式有效
            max_context_length: 最大上下文长度
            **kwargs: 额外参数
        """
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stream = stream
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.max_context_length = max_context_length
        self.kwargs = kwargs

        self.logger = setup_logger(self.__class__.__name__)

        # API client for HTTP requests (used by openai/aliyun/vllm providers)
        self._client = None

        # Local model components (used by local provider)
        self._model = None
        self._tokenizer = None

        # Initialize based on provider type
        if self.provider == "local":
            self._init_local_model()
        else:
            self._init_api_client()

    def _init_api_client(self) -> None:
        """初始化 API 客户端（用于 openai/aliyun/vllm 供应商）。"""
        if self.provider == "local":
            return

        if not self.model_name:
            raise ValueError(f"[{self.provider}] model_name is required for API providers")
        if not self.base_url:
            raise ValueError(f"[{self.provider}] base_url is required for API providers")
        if not self.api_key:
            raise ValueError(f"[{self.provider}] api_key is required for API providers")

        self.logger.info(f"Initializing {self.provider} LLM provider: {self.model_name}")
        self.logger.info(f"Base URL: {self.base_url}")

    def _init_local_model(self) -> None:
        """初始化本地 transformers 模型。"""
        if not self.model_path:
            raise ValueError("model_path is required for local provider")

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        self.logger.info(f"Loading local model from: {self.model_path}")

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if self.device != "cpu" else None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
        )

        if self.device == "cpu":
            self._model = self._model.to(self.device)

        self.logger.info("Local model loaded successfully")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        生成文本。

        Args:
            prompt: 输入提示词（字符串或消息列表）
            max_tokens: 最大生成长度
            temperature: 生成温度
            top_p: 核采样参数
            top_k: top-k 采样参数
            stop: 停止词列表
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        if self.provider == "local":
            return self._generate_local(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs,
            )
        else:
            return self._generate_api(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                **kwargs,
            )

    def _generate_api(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """通过 API 生成文本（openai/aliyun/vllm 供应商）。"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k

        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": self.stream,
        }

        if top_k != 50:  # Only include if non-default
            payload["top_k"] = top_k

        if stop:
            payload["stop"] = stop

        # Extra body for specific models (e.g., qwen)
        extra_body = kwargs.get("extra_body")
        if extra_body:
            payload["extra_body"] = extra_body

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # For some providers, use API key directly without Bearer prefix
        if self.provider in ("aliyun",):
            headers["Authorization"] = self.api_key

        try:
            self.logger.info(f"[{self.provider}] Sending request to {self.base_url}")
            self.logger.info(f"[{self.provider}] Model: {self.model_name}, messages count: {len(messages)}")

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=min(self.timeout, 200),
            )

            if response.status_code != 200:
                self.logger.error(f"[{self.provider}] Response status: {response.status_code}")
                self.logger.error(f"[{self.provider}] Response body: {response.text}")
                response.raise_for_status()

            response_data = response.json()

            if "error" in response_data:
                raise ValueError(f"[{self.provider}] API Error: {response_data['error']}")

            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError(f"[{self.provider}] Empty choices in response: {response_data}")

            choice = response_data["choices"][0]

            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]

            if "delta" in choice and "content" in choice["delta"]:
                return choice["delta"]["content"]

            raise ValueError(f"[{self.provider}] No content in response: {response_data}")

        except requests.exceptions.Timeout:
            raise TimeoutError(f"[{self.provider}] API request timeout ({self.timeout}s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[{self.provider}] API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"[{self.provider}] Generation error: {e}")

    def _generate_local(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """使用本地模型生成文本。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Local model not loaded. Check initialization.")

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k

        use_chat_template = hasattr(self._tokenizer, "apply_chat_template") and isinstance(prompt, list)

        if use_chat_template and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt
        ):
            messages = prompt
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, device=self.device),
            }
        else:
            if isinstance(prompt, list):
                prompt = "\n".join(prompt)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context_length,
            ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"],
                eos_token_id=self._tokenizer.eos_token_id,
                **kwargs,
            )

        if use_chat_template:
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant response after the last user message
            if "assistant" in response.lower():
                parts = response.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    if response.startswith(":"):
                        response = response[1:].strip()
        else:
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if isinstance(prompt, str) and response.startswith(prompt):
                response = response[len(prompt):].strip()

        return response

    def prepare_inputs_for_rag(
        self, query: str, context: List[str]
    ) -> Dict[str, Any]:
        """
        准备 RAG 流程的输入。

        Args:
            query: 用户查询
            context: 检索到的上下文文档列表

        Returns:
            Dict containing prepared inputs
        """
        context_str = "\n\n".join(
            [f"[Document {i+1}]: {doc}" for i, doc in enumerate(context)]
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant that provides accurate information based on retrieved medical documents.",
            },
            {
                "role": "user",
                "content": f"I need information about: {query}\n\nRelevant documents:\n{context_str}",
            },
        ]

        return {
            "messages": messages,
            "query": query,
            "context": context,
        }

    @property
    def model(self):
        """获取本地模型实例。"""
        return self._model

    @property
    def tokenizer(self):
        """获取 tokenizer 实例。"""
        return self._tokenizer

    def __repr__(self) -> str:
        return (
            f"LLMProvider(provider={self.provider}, model_name={self.model_name}, "
            f"model_path={self.model_path}, device={self.device})"
        )
