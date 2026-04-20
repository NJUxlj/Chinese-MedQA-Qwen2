"""
Embedding Provider - 统一的文本嵌入向量调用接口

支持以下模式:
- vllm: vLLM 部署的嵌入模型 API (model_name, base_url, api_key)
- local: 本地 sentence-transformers 模型 (model_name 即为模型名称或路径)
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import List, Union, Optional, Any, Literal
from pathlib import Path

import requests

# Add project root to path for utils import
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    统一 Embedding Provider，支持 vLLM API 和本地 sentence-transformers 模型。

    模式 (mode):
        - vllm: vLLM 部署的嵌入模型 API
        - local: 本地 sentence-transformers 模型
    """

    def __init__(
        self,
        mode: Literal["vllm", "local"] = "local",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        encode_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        初始化 Embedding Provider。

        Args:
            mode: 运行模式，支持 vllm/local
            model_name: 模型名称（vLLM 模式必填）
            base_url: API 基础 URL（vLLM 模式必填）
            api_key: API 密钥（vLLM 模式可选）
            model_path: 本地模型路径（local 模式可选，默认使用 model_name）
            device: 运行设备（cuda/cpu/mps），仅 local 模式有效
            normalize_embeddings: 是否归一化嵌入向量
            batch_size: 批处理大小
            encode_kwargs: 额外编码参数
            **kwargs: 额外参数
        """
        self.mode = mode
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.encode_kwargs = encode_kwargs or {}
        self.kwargs = kwargs

        self.logger = setup_logger(self.__class__.__name__)

        # Local embedding model (sentence-transformers)
        self._model = None

        # Initialize
        if self.mode == "local":
            self._init_local_model()
        elif self.mode == "vllm":
            self._init_vllm_client()
        else:
            raise ValueError(f"Unknown embedding mode: {mode}. Supported: vllm, local")

    def _init_local_model(self) -> None:
        """初始化本地 sentence-transformers 模型。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding mode. "
                "Install it with: pip install sentence-transformers"
            )

        model_name_or_path = self.model_path or self.model_name

        if not model_name_or_path:
            raise ValueError("model_name or model_path is required for local embedding mode")

        self.logger.info(f"Loading sentence-transformers model: {model_name_or_path}")

        self._model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=self.device,
            **self.kwargs,
        )

        self.logger.info("Local embedding model loaded successfully")

    def _init_vllm_client(self) -> None:
        """初始化 vLLM API 客户端。"""
        if not self.model_name:
            raise ValueError("model_name is required for vLLM embedding mode")
        if not self.base_url:
            raise ValueError("base_url is required for vLLM embedding mode")

        self.logger.info(f"Initializing vLLM embedding client: {self.model_name}")
        self.logger.info(f"Base URL: {self.base_url}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        编码文本为嵌入向量。

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            normalize_embeddings: 是否归一化嵌入向量
            **kwargs: 额外参数

        Returns:
            np.ndarray: 嵌入向量数组，shape 为 (n, embedding_dim)
        """
        if self.mode == "local":
            return self._encode_local(
                texts,
                batch_size=batch_size,
                show_progress=show_progress,
                normalize_embeddings=normalize_embeddings,
                **kwargs,
            )
        else:
            return self._encode_vllm(texts, normalize_embeddings=normalize_embeddings, **kwargs)

    def _encode_local(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """使用 sentence-transformers 编码文本。"""
        if self._model is None:
            raise RuntimeError("Local embedding model not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size
        normalize = (
            normalize_embeddings
            if normalize_embeddings is not None
            else self.normalize_embeddings
        )

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            **self.encode_kwargs,
            **kwargs,
        )

        return embeddings

    def _encode_vllm(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """通过 vLLM API 编码文本。"""
        if isinstance(texts, str):
            texts = [texts]

        normalize = (
            normalize_embeddings
            if normalize_embeddings is not None
            else self.normalize_embeddings
        )

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "input": texts,
            "normalize_embeddings": normalize,
            **kwargs,
        }

        try:
            self.logger.info(f"[vLLM] Encoding {len(texts)} texts")
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                self.logger.error(f"[vLLM] Response status: {response.status_code}")
                self.logger.error(f"[vLLM] Response body: {response.text}")
                response.raise_for_status()

            response_data = response.json()

            if "error" in response_data:
                raise ValueError(f"[vLLM] API Error: {response_data['error']}")

            if "data" not in response_data or not response_data["data"]:
                raise ValueError(f"[vLLM] No embeddings in response: {response_data}")

            # Sort by index to maintain order
            embeddings = sorted(response_data["data"], key=lambda x: x["index"])
            embeddings_array = np.array([item["embedding"] for item in embeddings])

            return embeddings_array

        except requests.exceptions.Timeout:
            raise TimeoutError("[vLLM] Embedding request timeout")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[vLLM] Embedding request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"[vLLM] Encoding error: {e}")

    def embed_query(self, text: str, **kwargs) -> np.ndarray:
        """
        编码单个查询文本（query）。

        Args:
            text: 查询文本
            **kwargs: 额外参数

        Returns:
            np.ndarray: 嵌入向量
        """
        embedding = self.encode([text], **kwargs)
        return embedding[0]

    def embed_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        批量编码文档文本。

        Args:
            texts: 文档文本列表
            **kwargs: 额外参数

        Returns:
            np.ndarray: 嵌入向量数组
        """
        return self.encode(texts, **kwargs)

    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量的维度。"""
        if self.mode == "local" and self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        else:
            raise NotImplementedError(
                f"embedding_dim is only available for local mode after model is loaded"
            )

    def __repr__(self) -> str:
        return (
            f"EmbeddingProvider(mode={self.mode}, model_name={self.model_name}, "
            f"model_path={self.model_path}, device={self.device})"
        )
