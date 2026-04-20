"""
LLM Provider - 统一的大语言模型调用接口

支持以下供应商:
- openai: OpenAI 兼容 API (model_name, base_url, api_key)
- aliyun: 阿里云百炼 API (model_name, base_url, api_key)
- vllm: vLLM 部署的模型 API (model_name, base_url, api_key)
- local: 本地模型推理，支持 transformers / vllm / ollama 三种后端
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
        - local: 本地模型推理（后端由 local_backend 指定）

    本地后端类型 (local_backend):
        - transformers: 使用 transformers 库加载本地模型权重
        - vllm: 使用 vLLM 引擎加载本地模型
        - ollama: 连接到本地 Ollama 服务（需提前启动 ollama run）
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
        # local 模式专用参数
        local_backend: Literal["transformers", "vllm", "ollama"] = "transformers",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        ollama_host: str = "http://localhost:11434",
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
            device: 运行设备（cuda/cpu/mps），仅 transformers 后端有效
            load_in_8bit: 是否量化到 8bit，仅 transformers 后端有效
            load_in_4bit: 是否量化到 4bit，仅 transformers 后端有效
            trust_remote_code: 是否信任远程代码
            max_context_length: 最大上下文长度
            local_backend: 本地推理后端，transformers / vllm / ollama
            tensor_parallel_size: 张量并行大小，仅 vllm 后端有效
            gpu_memory_utilization: GPU 显存使用率，仅 vllm 后端有效
            ollama_host: Ollama 服务地址，仅 ollama 后端有效
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
        self.local_backend = local_backend
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.ollama_host = ollama_host
        self.kwargs = kwargs

        self.logger = setup_logger(self.__class__.__name__)

        # Local backend model components
        self._model = None       # transformers/vllm 模型实例
        self._tokenizer = None   # transformers/ollama tokenizer
        self._llm = None         # vLLM LLM 实例
        self._ollama_client = None  # ollama.Client 实例

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
        """
        初始化本地模型，根据 local_backend 选择对应后端。

        支持的后端:
            - transformers: 使用 transformers 库加载本地模型权重
            - vllm: 使用 vLLM 引擎加载本地模型（高吞吐量）
            - ollama: 连接到本地 Ollama 服务（需提前启动 ollama run）
        """
        if not self.model_path:
            raise ValueError("model_path is required for local provider")

        if self.local_backend == "transformers":
            self._init_transformers()
        elif self.local_backend == "vllm":
            self._init_vllm()
        elif self.local_backend == "ollama":
            self._init_ollama()
        else:
            raise ValueError(
                f"Unknown local_backend: {self.local_backend}. "
                "Supported: transformers, vllm, ollama"
            )

    def _init_transformers(self) -> None:
        """使用 transformers 库加载本地模型。"""
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        self.logger.info(f"[transformers] Loading model from: {self.model_path}")

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

        self.logger.info("[transformers] Model loaded successfully")

    def _init_vllm(self) -> None:
        """
        使用 vLLM 引擎加载本地模型。

        vLLM 支持:
        - PagedAttention 高效显存管理
        - 张量并行 (tensor_parallel_size > 1)
        - 量化 (awq/gptq/squeezellm)
        - 流式生成

        安装: pip install vllm
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "[vllm] vLLM is not installed. "
                "Install with: pip install vllm"
            )

        self.logger.info(f"[vllm] Loading model from: {self.model_path}")
        self.logger.info(f"[vllm] tensor_parallel_size={self.tensor_parallel_size}, "
                         f"gpu_memory_utilization={self.gpu_memory_utilization}")

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count < self.tensor_parallel_size:
            self.logger.warning(
                f"[vllm] Available GPUs ({gpu_count}) < requested tensor_parallel_size "
                f"({self.tensor_parallel_size}), using {max(1, gpu_count)}"
            )
            self.tensor_parallel_size = max(1, gpu_count)

        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            max_model_len=self.max_context_length,
            dtype="auto",
        )

        self.logger.info("[vllm] Model loaded successfully")

    def _init_ollama(self) -> None:
        """
        连接到本地 Ollama 服务进行推理。

        Ollama 特点:
        - 模型需提前通过 `ollama run <model>` 或 `ollama pull <model>` 下载
        - 服务默认运行在 http://localhost:11434
        - 支持流式输出和对话模板

        安装 Ollama: https://ollama.ai
        下载模型: ollama pull qwen2.5
        启动服务: ollama serve  (通常自动启动)
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "[ollama] ollama Python client is not installed. "
                "Install with: pip install ollama"
            )

        self.logger.info(f"[ollama] Connecting to Ollama at: {self.ollama_host}")
        self.logger.info(f"[ollama] Model name: {self.model_name or self.model_path}")

        # model_name 用于 ollama API，model_path 作为 fallback 显示名
        model_identifier = self.model_name or self.model_path

        # 验证服务可连接
        try:
            client = ollama.Client(host=self.ollama_host)
            models_response = client.list()
            self.logger.info(f"[ollama] Connected. Available models: {models_response}")

            # 检查模型是否已存在
            available_names = []
            if hasattr(models_response, 'models'):
                available_names = [m.get('name', '') for m in models_response.models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                available_names = [m.get('name', '') for m in models_response.get('models', [])]

            if model_identifier not in available_names:
                self.logger.warning(
                    f"[ollama] Model '{model_identifier}' not found in available models. "
                    f"Available: {available_names}. "
                    f"Pull it with: ollama pull {model_identifier}"
                )
        except Exception as e:
            self.logger.warning(f"[ollama] Could not connect to Ollama: {e}")

        self._ollama_client = client
        # model_name 存储 ollama 使用的模型名
        self.model_name = model_identifier
        self.logger.info(f"[ollama] Ollama client initialized for model: {self.model_name}")

    def _generate_local(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """使用本地模型生成文本，根据后端分发到对应生成逻辑。"""
        if self.local_backend == "transformers":
            return self._generate_transformers(
                prompt, max_tokens, temperature, top_p, top_k, **kwargs
            )
        elif self.local_backend == "vllm":
            return self._generate_vllm(
                prompt, max_tokens, temperature, top_p, top_k, **kwargs
            )
        elif self.local_backend == "ollama":
            return self._generate_ollama(
                prompt, max_tokens, temperature, top_p, top_k, **kwargs
            )
        else:
            raise ValueError(f"Unknown local_backend: {self.local_backend}")

    def _generate_transformers(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """使用 transformers 模型生成文本。"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Transformers model not loaded.")

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k

        use_chat_template = (
            hasattr(self._tokenizer, "apply_chat_template")
            and isinstance(prompt, list)
        )

        if use_chat_template and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in prompt
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

    def _generate_vllm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """使用 vLLM 引擎生成文本。"""
        if self._llm is None:
            raise RuntimeError("vLLM model not loaded.")

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        stop = kwargs.get("stop")

        # vLLM 的 SamplingParams
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0 if temperature == 0 else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=-1 if top_k is None else top_k,
            stop=stop or ["<|im_end|>"],
            echo=False,
        )

        # 处理 prompt 格式
        if isinstance(prompt, list) and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in prompt
        ):
            # 使用 chat template
            prompt_text = self._build_chat_prompt(prompt)
        elif isinstance(prompt, list):
            prompt_text = "\n".join(prompt)
        else:
            prompt_text = prompt

        outputs = self._llm.generate(prompt_text, sampling_params)
        return outputs[0].outputs[0].text

    def _generate_ollama(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """使用 Ollama 服务生成文本。"""
        import ollama

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        # 处理 prompt 格式
        if isinstance(prompt, list) and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in prompt
        ):
            # Ollama chat 模式
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in prompt
            ]
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": kwargs.get("stop"),
                },
                stream=False,
            )
            return response["message"]["content"]
        else:
            # Ollama generate 模式
            prompt_text = prompt if isinstance(prompt, str) else "\n".join(prompt)
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt_text,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": kwargs.get("stop"),
                },
                stream=False,
            )
            return response["response"]

    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表构建为纯文本 prompt（用于不支持 chat template 的后端）。"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

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
                stop=stop,
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
        elif isinstance(prompt, list) and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in prompt
        ):
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

        if top_k != 50:
            payload["top_k"] = top_k

        if stop:
            payload["stop"] = stop

        extra_body = kwargs.get("extra_body")
        if extra_body:
            payload["extra_body"] = extra_body

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if self.provider in ("aliyun",):
            headers["Authorization"] = self.api_key

        try:
            self.logger.info(f"[{self.provider}] Sending request to {self.base_url}")
            self.logger.info(f"[{self.provider}] Model: {self.model_name}")

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
        return self._model or self._llm

    @property
    def tokenizer(self):
        """获取 tokenizer 实例。"""
        return self._tokenizer

    def __repr__(self) -> str:
        if self.provider == "local":
            return (
                f"LLMProvider(provider=local, backend={self.local_backend}, "
                f"model_path={self.model_path}, device={self.device})"
            )
        return (
            f"LLMProvider(provider={self.provider}, model_name={self.model_name}, "
            f"base_url={self.base_url})"
        )
