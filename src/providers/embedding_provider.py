"""
Embedding Provider - 统一的文本嵌入向量调用接口

支持以下模式:
- vllm: vLLM 部署的嵌入模型 API (model_name, base_url, api_key)
- local: 本地 sentence-transformers 模型 (model_name 即为模型名称或路径)
- openai: OpenAI Embeddings API
- bge: HuggingFace BGE Embeddings (LangChain)
- modelscope: ModelScope Embeddings (LangChain)
- huggingface: HuggingFace Embeddings (LangChain)
"""

import os
import sys
import torch
import numpy as np
import logging
import pickle
import time
from typing import List, Union, Optional, Any, Literal, Dict
from pathlib import Path

import requests

# Add project root to path for utils import
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    统一 Embedding Provider，支持多种 embedding 后端和缓存机制。

    支持的模式 (mode):
        - vllm: vLLM 部署的嵌入模型 API
        - local: 本地 sentence-transformers 模型
        - openai: OpenAI Embeddings API
        - bge: HuggingFace BGE Embeddings
        - modelscope: ModelScope Embeddings
        - huggingface: HuggingFace Embeddings
    """

    def __init__(
        self,
        mode: Literal["vllm", "local", "openai", "bge", "modelscope", "huggingface"] = "local",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        encode_kwargs: Optional[dict] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        初始化 Embedding Provider。

        Args:
            mode: 运行模式，支持 vllm/local/openai/bge/modelscope/huggingface
            model_name: 模型名称
            base_url: API 基础 URL（vLLM 模式必填）
            api_key: API 密钥（vLLM/OpenAI 模式可选）
            model_path: 本地模型路径（local 模式可选，默认使用 model_name）
            device: 运行设备（cuda/cpu/mps），仅 local 模式有效
            normalize_embeddings: 是否归一化嵌入向量
            batch_size: 批处理大小
            encode_kwargs: 额外编码参数
            cache_dir: 缓存目录路径（可选）
            use_cache: 是否使用磁盘缓存
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
        self.use_cache = use_cache
        self.embedding_model_name = model_name

        # Cache settings
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        self.embedding_cache: Dict[str, List[float]] = {}

        self.logger = setup_logger(self.__class__.__name__)

        # Local embedding model (sentence-transformers)
        self._model = None
        # LangChain embeddings model (for openai/bge/modelscope/huggingface)
        self._lc_embeddings = None

        # Initialize
        if self.mode == "local":
            self._init_local_model()
        elif self.mode == "vllm":
            self._init_vllm_client()
        elif self.mode in ["openai", "bge", "modelscope", "huggingface"]:
            self._init_langchain_embeddings()
        else:
            raise ValueError(f"Unknown embedding mode: {mode}. Supported: vllm, local, openai, bge, modelscope, huggingface")

        # Load cache if enabled
        if self.use_cache and self.cache_dir:
            self._load_cache()

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

    def _init_langchain_embeddings(self) -> None:
        """初始化 LangChain Embeddings 模型（openai/bge/modelscope/huggingface）。"""
        if not self.model_name:
            raise ValueError("model_name is required for LangChain embedding mode")

        self.logger.info(f"Loading LangChain embedding model: {self.model_name} (mode: {self.mode})")

        try:
            if self.mode == "openai":
                if not self.api_key:
                    raise ValueError("API key is required for OpenAI embeddings mode")
                from langchain_openai import OpenAIEmbeddings
                self._lc_embeddings = OpenAIEmbeddings(model=self.model_name, openai_api_key=self.api_key)

            elif self.mode == "bge":
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                self._lc_embeddings = HuggingFaceBgeEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={"normalize_embeddings": self.normalize_embeddings}
                )

            elif self.mode == "modelscope":
                from langchain_community.embeddings import ModelScopeEmbeddings
                self._lc_embeddings = ModelScopeEmbeddings(model_name=self.model_name)

            elif self.mode == "huggingface":
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._lc_embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={"normalize_embeddings": self.normalize_embeddings}
                )

            self.logger.info(f"LangChain embedding model loaded successfully in {self.mode} mode")
        except ImportError as e:
            raise ImportError(f"Failed to load {self.mode} embeddings. Required package may not be installed: {e}")

    def _get_cache_path(self) -> Path:
        """获取缓存文件路径。"""
        if not self.cache_dir:
            raise ValueError("Cache directory not set")
        model_name_safe = self.embedding_model_name.replace("/", "_") if self.embedding_model_name else "unknown"
        return self.cache_dir / f"{model_name_safe}_embedding_cache.pkl"

    def _load_cache(self) -> None:
        """从磁盘加载 embedding 缓存。"""
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings from {cache_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding cache: {str(e)}")
                self.embedding_cache = {}

    def _save_cache(self) -> None:
        """保存 embedding 缓存到磁盘。"""
        if not self.use_cache or not self.cache_dir:
            return
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
            self.logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache at {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {str(e)}")

    def _get_cache_key(self, text: str) -> str:
        """生成文本的缓存键。"""
        return str(hash(text))

    def _encode_langchain(self, texts: List[str]) -> List[List[float]]:
        """使用 LangChain Embeddings 编码文本。"""
        if self._lc_embeddings is None:
            raise RuntimeError("LangChain embedding model not loaded")
        return self._lc_embeddings.embed_documents(texts)

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
        elif self.mode == "vllm":
            return self._encode_vllm(texts, normalize_embeddings=normalize_embeddings, **kwargs)
        else:
            # LangChain modes (openai, bge, modelscope, huggingface)
            return self._encode_langchain_arrays(texts, **kwargs)

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

    def _encode_langchain_arrays(self, texts: List[str], **kwargs) -> np.ndarray:
        """使用 LangChain Embeddings 编码文本，返回 np.ndarray。"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._encode_langchain(texts)
        return np.array(embeddings)

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """
        编码单个查询文本（query）。

        Args:
            text: 查询文本
            **kwargs: 额外参数

        Returns:
            List[float]: 嵌入向量
        """
        if not text.strip():
            raise ValueError("Query cannot be empty")

        cache_key = self._get_cache_key(text)

        # Check cache first
        if self.use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Get embedding
        try:
            if self.mode == "local":
                embedding = self._encode_local([text], **kwargs)[0].tolist()
            elif self.mode == "vllm":
                embedding = self._encode_vllm([text], **kwargs)[0].tolist()
            else:
                embedding = self._lc_embeddings.embed_query(text)

            # Cache the result
            if self.use_cache:
                self.embedding_cache[cache_key] = embedding
                self._save_cache()

            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding query: {str(e)}")
            raise

    def embed_documents(self, documents: List[Any], **kwargs) -> Dict[str, List[float]]:
        """
        批量编码文档文本。

        Args:
            documents: 文档列表（支持 Document 对象或字符串）
            **kwargs: 额外参数

        Returns:
            Dict[str, List[float]]: 字典，key 为 doc_id，value 为嵌入向量
        """
        if not documents:
            return {}

        # Extract texts and doc_ids
        texts = []
        doc_ids = []
        for i, doc in enumerate(documents):
            if hasattr(doc, 'page_content'):
                # Document object
                text = doc.page_content
                doc_id = str(doc.metadata.get("doc_id", str(i)) if hasattr(doc, 'metadata') else str(i))
            else:
                # Plain string
                text = doc
                doc_id = str(i)

            if text.strip():
                texts.append(text)
                doc_ids.append(doc_id)

        if not texts:
            return {}

        # Check cache for each text
        embeddings: Dict[str, List[float]] = {}
        texts_to_embed = []
        uncached_indices = []

        if self.use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embeddings[doc_ids[i]] = self.embedding_cache[cache_key]
                else:
                    texts_to_embed.append(text)
                    uncached_indices.append(i)
        else:
            texts_to_embed = texts
            uncached_indices = list(range(len(texts)))

        # Embed uncached texts
        if texts_to_embed:
            try:
                start_time = time.time()

                if self.mode == "local":
                    uncached_embeddings = self._encode_local(texts_to_embed, **kwargs)
                    uncached_embeddings = uncached_embeddings.tolist()
                elif self.mode == "vllm":
                    uncached_embeddings = self._encode_vllm(texts_to_embed, **kwargs).tolist()
                else:
                    uncached_embeddings = self._encode_langchain(texts_to_embed)

                self.logger.info(f"Embedded {len(texts_to_embed)} documents in {time.time()-start_time:.2f}s")

                # Cache the results
                for idx, text, embedding in zip(uncached_indices, texts_to_embed, uncached_embeddings):
                    cache_key = self._get_cache_key(text)
                    if self.use_cache:
                        self.embedding_cache[cache_key] = embedding
                    doc_id = doc_ids[idx]
                    embeddings[doc_id] = embedding

            except Exception as e:
                self.logger.error(f"Error embedding documents: {str(e)}")
                raise

        # Save cache after embedding
        if self.use_cache:
            self._save_cache()

        return embeddings

    def batch_embed_texts(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> List[List[float]]:
        """
        批量编码文本（带缓存）。

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            **kwargs: 额外参数

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        result = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                batch_embeddings = []
                texts_to_embed = []
                uncached_indices = []

                if self.use_cache:
                    for j, text in enumerate(batch):
                        cache_key = self._get_cache_key(text)
                        if cache_key in self.embedding_cache:
                            batch_embeddings.append((j, self.embedding_cache[cache_key]))
                        else:
                            texts_to_embed.append(text)
                            uncached_indices.append(j)
                else:
                    texts_to_embed = batch
                    uncached_indices = list(range(len(batch)))

                # Embed uncached texts
                if texts_to_embed:
                    if self.mode == "local":
                        uncached_embeddings = self._encode_local(texts_to_embed, **kwargs).tolist()
                    elif self.mode == "vllm":
                        uncached_embeddings = self._encode_vllm(texts_to_embed, **kwargs).tolist()
                    else:
                        uncached_embeddings = self._encode_langchain(texts_to_embed)

                    # Cache the results
                    if self.use_cache:
                        for j, text, embedding in zip(uncached_indices, texts_to_embed, uncached_embeddings):
                            cache_key = self._get_cache_key(text)
                            self.embedding_cache[cache_key] = embedding
                            batch_embeddings.append((j, embedding))
                    else:
                        batch_embeddings.extend(zip(uncached_indices, uncached_embeddings))

                # Sort embeddings by original indices
                batch_embeddings.sort(key=lambda x: x[0])
                result.extend([emb for _, emb in batch_embeddings])

                self.logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            except Exception as e:
                self.logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                raise

        return result

    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量的维度。"""
        if self.mode == "local" and self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        elif self._lc_embeddings is not None:
            # Try to get dimension from LangChain embeddings
            try:
                test_emb = self._lc_embeddings.embed_query("test")
                return len(test_emb)
            except Exception:
                pass
            raise NotImplementedError(
                f"embedding_dim is only available for local mode after model is loaded, "
                f"or for LangChain modes after a query has been made"
            )
        else:
            raise NotImplementedError(
                f"embedding_dim is only available for local mode after model is loaded"
            )

    def __repr__(self) -> str:
        return (
            f"EmbeddingProvider(mode={self.mode}, model_name={self.model_name}, "
            f"model_path={self.model_path}, device={self.device})"
        )
