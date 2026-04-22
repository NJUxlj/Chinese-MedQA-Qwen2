from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import torch
import logging
import requests
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import settings
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class RerankerProvider:
    """ reranker 服务

    使用 reranker 模型，根据对 query 的相似度， 对已有的 document 列表进行重排序

        - 用于从本地加载 Qwen3-reranker 这样的模型进行重排序
        - Qwen3-Reranker 使用特殊的输入格式和输出机制
        - 根据模型名称自动选择使用传统 CrossEncoder 方式、Qwen3-Reranker 特殊方式、或 vLLM API 方式
        - 优先级：若配置了 base_url，则优先使用 vLLM API 方式；否则根据 model_provider 决定本地加载方式
    """
    def __init__(self, config=None):
        """初始化 reranker 服务"""
        self.config = config if config is not None else settings.reranker
        self.model_provider = self.config.model_provider
        self.reranker_model = None
        self.tokenizer = None
        self.is_qwen3_reranker = "qwen3" in self.config.model_name.lower()
        self.is_vllm = bool(self.config.base_url)  # 优先判断：如果有 base_url 则使用 vLLM API
        self.true_token_id = None
        self.false_token_id = None
        self._load_reranker_model()


    def _load_reranker_model(self) -> None:
        """加载 reranker 模型"""
        # 优先判断：如果配置了 base_url，则使用 vLLM API 方式
        if self.is_vllm:
            logger.info(f"Using vLLM API reranker: {self.config.base_url}")
            logger.info(f"Reranker model name: {self.config.model_name}")
            # vLLM 模式不需要加载本地模型，只需验证配置
            if not self.config.model_name:
                raise ValueError("model_name is required when using vLLM API reranker")
            return

        model_path = self.config.model_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading reranker model from: {model_path}")
        logger.info(f"Model name: {self.config.model_name}, Is Qwen3-Reranker: {self.is_qwen3_reranker}")

        if self.is_qwen3_reranker:
            logger.info("Using transformers AutoModelForCausalLM for Qwen3-Reranker")
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.config.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self._setup_qwen3_reranker_tokens()
            logger.info(f"Successfully loaded Qwen3-Reranker model: {self.config.model_name}")
        else:
            if self.model_provider == "sentence_transformers":
                logger.info("Using sentence_transformers CrossEncoder")
                self.reranker_model = CrossEncoder(
                    model_name=model_path,
                    device=self.config.device,
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded CrossEncoder model: {self.config.model_name}")
            else:
                raise ValueError(f"Invalid model provider: {self.model_provider}. Currently only 'sentence_transformers' is supported.")

    def _setup_qwen3_reranker_tokens(self) -> None:
        """设置 Qwen3-Reranker 的 yes/no token ID"""
        try:
            tokenizer = self.tokenizer

            yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
            no_tokens = tokenizer.encode("no", add_special_tokens=False)

            if yes_tokens:
                self.true_token_id = yes_tokens[0]
            if no_tokens:
                self.false_token_id = no_tokens[0]

            logger.info(f"Qwen3-Reranker token IDs - 'yes': {self.true_token_id}, 'no': {self.false_token_id}")

            if self.true_token_id is None or self.false_token_id is None:
                logger.warning("Could not find 'yes' or 'no' token IDs. Output processing may not work correctly.")

        except Exception as e:
            logger.error(f"Error setting up Qwen3-Reranker tokens: {str(e)}")


    def _format_query_document_pairs_for_qwen3(self,query: str, prefix, suffix, documents: List[Document], query_domain = "medical") -> List[str]:
        """格式化 query-document 对为 Qwen3-Reranker 所需的输入格式（使用聊天模板）

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            格式化后的文本列表
        """
        formatted_pairs = []


        instruction = f'Given a {query_domain} search query, retrieve relevant passages that answer the query'
        basic_format = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

        for doc in documents:
            basic_formatted_text = basic_format.format(
                instruction=instruction,
                query=query,
                doc=doc.page_content
            )

            formatted_text = prefix + basic_formatted_text + suffix

            formatted_pairs.append(formatted_text)

        return formatted_pairs


    def _process_qwen3_reranker_scores(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """处理 Qwen3-Reranker 的原始输出，获取相关性分数

        Qwen3-Reranker 的输出是一个 logits 向量，对应 "yes" 和 "no" 的分数
        我们需要计算 "yes" 的概率作为相关性分数

        Args:
            raw_outputs: 模型的原始输出 (batch_size, vocab_size)  【取每个序列的最后一个 token】

        Returns:
            相关性分数 (batch_size,)
        """
        if not isinstance(raw_outputs, torch.Tensor):
            raw_outputs = torch.tensor(raw_outputs)

        if raw_outputs.dim() == 1:
            raw_outputs = raw_outputs.unsqueeze(0)

        if self.true_token_id is not None and self.false_token_id is not None:
            true_scores = raw_outputs[:, self.true_token_id]
            false_scores = raw_outputs[:, self.false_token_id]

            scores = torch.stack([false_scores, true_scores], dim=1)
            scores = torch.nn.functional.softmax(scores, dim=1)
            relevance_scores = scores[:, 1]

            return relevance_scores.squeeze()
        else:
            logger.warning("True/False token IDs not found. Using raw output mean as score.")
            return raw_outputs.mean(dim=1)


    def _format_query_document_pairs(self, query: str, documents: List[Document]) -> List[str]:
        """格式化 query-document 对为传统 CrossEncoder 所需的输入格式

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            格式化后的文本对列表
        """
        formatted_pairs = []
        for doc in documents:
            formatted_text = [query, doc.page_content]
            formatted_pairs.append(formatted_text)

        return formatted_pairs


    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """ rerank 文档

        Args:
            query: 查询字符串
            documents: 待重排序的文档列表
            top_k: 返回的 top_k 个文档

        Returns:
            重排序后的文档列表
        """
        if not query:
            logger.warning("Query is empty, returning original documents")
            return documents[:top_k]

        if not documents:
            logger.warning("Documents list is empty")
            return []

        if self.is_vllm:
            return self._rerank_with_vllm_api(query, documents, top_k)

        if self.reranker_model is None:
            raise ValueError("Reranker model not loaded")

        try:
            logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
            logger.info(f"Using {'Qwen3-Reranker (transformers)' if self.is_qwen3_reranker else 'Traditional CrossEncoder'} approach")

            if self.is_qwen3_reranker:
                scores = self._rerank_with_qwen3_transformers(query, documents)
            else:
                scores = self._rerank_with_cross_encoder(query, documents)

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            if self.config.normalize_scores:
                min_score = scores.min()
                max_score = scores.max()
                if max_score - min_score > 0:
                    scores = (scores - min_score) / (max_score - min_score)
                else:
                    scores = scores - min_score

            doc_score_pairs = list(zip(documents, scores))
            new_doc_score_pairs = []
            for doc, score in doc_score_pairs:
                new_meta_data = doc.metadata.copy() if doc.metadata else {}
                new_meta_data["rerank_score"] = float(score)
                doc.metadata = new_meta_data
                new_doc_score_pairs.append((doc, float(score)))

            new_doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            reranked_documents = [doc for doc, _ in new_doc_score_pairs[:top_k]]

            logger.info(f"Reranking completed. Top {top_k} documents selected.")

            self.print_reranked_documents_and_scores(new_doc_score_pairs[:top_k])

            return reranked_documents

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            raise


    def _rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> torch.Tensor:
        """使用传统 CrossEncoder 方式进行重排序

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            相关性分数
        """
        formatted_inputs = self._format_query_document_pairs(query, documents)

        scores = self.reranker_model.predict(
            formatted_inputs,
            batch_size=self.config.batch_size
        )

        return scores


    def _rerank_with_vllm_api(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """使用 vLLM API 方式进行重排序

        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回的 top_k 个文档

        Returns:
            重排序后的文档列表
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        formatted_pairs = []
        for doc in documents:
            # 构建符合 Qwen3-Reranker 格式的输入
            instruction = "Given a medical search query, retrieve relevant passages that answer the query"
            formatted_text = (
                f"<|im_start|>system\n"
                f"Judge whether the Document meets the requirements based on the Query and the "
                f"Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<Instruct>: {instruction}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc.page_content}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n\n"
            )
            formatted_pairs.append(formatted_text)

        all_scores = []
        for i in range(0, len(formatted_pairs), self.config.batch_size):
            batch = formatted_pairs[i:i + self.config.batch_size]
            payload = {
                "prompt": batch,
                "model": self.config.model_name,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True
            }
            try:
                response = requests.post(
                    f"{self.config.base_url}/v1/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()

                # 从 logprobs 中提取 yes token 的概率作为分数
                for choice in result.get("choices", []):
                    logprobs = choice.get("logprobs", {})
                    tokens = logprobs.get("tokens", [])
                    token_logprobs = logprobs.get("token_logprobs", [])
                    # 查找 "yes" token 的位置并获取其 logprob
                    yes_score = 0.0
                    for j, token in enumerate(tokens):
                        if token.lower() == "yes" and j < len(token_logprobs):
                            yes_score = token_logprobs[j]
                            break
                    # 如果没找到 yes，使用 softmax 后的概率
                    if yes_score == 0.0:
                        # 简单处理：使用 0.5 作为默认值
                        yes_score = -0.693  # 约等于 0.5 的 logprob
                    all_scores.append(torch.tensor(yes_score, dtype=torch.float32))

            except Exception as e:
                logger.warning(f"vLLM API batch {i//self.config.batch_size} failed: {str(e)}")
                # 返回均匀分数
                all_scores.extend([torch.tensor(0.0) for _ in range(len(batch))])

        if not all_scores:
            logger.warning("No scores from vLLM API, returning original order")
            return documents[:top_k]

        scores = torch.stack(all_scores) if all_scores else torch.zeros(len(documents))

        # 归一化
        if self.config.normalize_scores and len(scores) > 0:
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                scores = (scores - min_score) / (max_score - min_score)
            else:
                scores = scores - min_score

        doc_score_pairs = list(zip(documents, scores.tolist()))
        new_doc_score_pairs = []
        for doc, score in doc_score_pairs:
            new_metadata = doc.metadata.copy() if doc.metadata else {}
            new_metadata["rerank_score"] = float(score)
            doc_copy = Document(page_content=doc.page_content, metadata=new_metadata)
            new_doc_score_pairs.append((doc_copy, float(score)))

        new_doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_documents = [doc for doc, _ in new_doc_score_pairs[:top_k]]

        logger.info(f"vLLM API reranking completed. Top {top_k} documents selected.")
        self.print_reranked_documents_and_scores(new_doc_score_pairs[:top_k])

        return reranked_documents


    def _rerank_with_qwen3(self, query: str, documents: List[Document]) -> torch.Tensor:
        """使用 Qwen3-Reranker 方式进行重排序（通过 CrossEncoder 预测）

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            相关性分数
        """
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        formatted_inputs = self._format_query_document_pairs_for_qwen3(query, prefix, suffix, documents)

        logger.debug(f"Number of documents to rank: {len(formatted_inputs)}")
        logger.debug(f"Formatted input example (first): {formatted_inputs[0][:200]}...")

        try:
            raw_outputs = self.reranker_model.predict(
                formatted_inputs,
                apply_softmax=True,
                convert_to_tensor=True
            )

            if isinstance(raw_outputs, torch.Tensor):
                if raw_outputs.dim() == 1:
                    raw_outputs = raw_outputs.unsqueeze(0)

                if raw_outputs.shape[-1] == 1:
                    scores = raw_outputs.squeeze(-1)
                elif raw_outputs.shape[-1] > 1:
                    if self.true_token_id is not None and self.true_token_id < raw_outputs.shape[-1]:
                        scores = raw_outputs[:, self.true_token_id].squeeze(-1)
                    else:
                        scores = torch.nn.functional.softmax(raw_outputs.float(), dim=-1)[:, 1].squeeze(-1)
                else:
                    scores = torch.zeros(raw_outputs.shape[0])
            else:
                if len(raw_outputs) == 0:
                    scores = torch.tensor([])
                elif raw_outputs.ndim == 1:
                    scores = torch.tensor(raw_outputs)
                else:
                    if self.true_token_id is not None and self.true_token_id < raw_outputs.shape[-1]:
                        scores = torch.tensor(raw_outputs[:, self.true_token_id])
                    else:
                        import numpy as np
                        scores = torch.tensor(np.argmax(raw_outputs, axis=1))

            return scores

        except Exception as e:
            logger.warning(f"Error in batch processing: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return torch.zeros(len(formatted_inputs))

    def _rerank_with_qwen3_transformers(self, query: str, documents: List[Document]) -> torch.Tensor:
        """使用 Qwen3-Reranker 方式进行重排序（通过 transformers 直接推理，批量处理）

        Args:
            query: 查询字符串
            documents: 文档列表

        Returns:
            相关性分数
        """
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n\n</think>\n\n"

        formatted_inputs = self._format_query_document_pairs_for_qwen3(query, prefix, suffix, documents)

        logger.debug(f"Number of documents to rank: {len(formatted_inputs)}")
        logger.debug(f"Formatted input example (first): {formatted_inputs[0][:200]}...")


        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens = False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens = False)

        try:
            inputs = self.tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=False,
                truncation='longest_first',
                max_length=2048 - len(prefix_tokens) - len(suffix_tokens)
            )

            input_ids = inputs["input_ids"].to(self.reranker_model.device)
            attention_mask = inputs["attention_mask"].to(self.reranker_model.device)

            with torch.no_grad():
                outputs = self.reranker_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )

            batch_logits = outputs.logits[:, -1, :]

            yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
            no_tokens = self.tokenizer.encode("no", add_special_tokens=False)

            if yes_tokens and no_tokens:
                yes_token_id = yes_tokens[0]
                no_token_id = no_tokens[0]

                true_vector = batch_logits[:, yes_token_id]
                false_vector = batch_logits[:, no_token_id]

                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp()

                return scores
            else:
                logger.warning("Could not find 'yes' or 'no' token IDs")
                return torch.zeros(len(formatted_inputs))

        except Exception as e:
            logger.warning(f"Error in batch processing: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return torch.zeros(len(formatted_inputs))


    def _process_single_qwen3_output(self, raw_output: Any) -> float:
        """处理单个 Qwen3-Reranker 输出

        Args:
            raw_output: 模型的原始输出

        Returns:
            相关性分数 (0-1)
        """
        try:
            if isinstance(raw_output, torch.Tensor):
                if raw_output.dim() == 0:
                    raw_output = raw_output.unsqueeze(0)

                if raw_output.shape[-1] > 1:
                    if self.true_token_id is not None and self.false_token_id is not None:
                        true_scores = raw_output[:, self.true_token_id]
                        false_scores = raw_output[:, self.false_token_id]

                        scores = torch.stack([false_scores, true_scores], dim=1)
                        scores = torch.nn.functional.softmax(scores, dim=1)
                        relevance_score = scores[:, 1].item()
                        return relevance_score

            if isinstance(raw_output, (list, torch.Tensor)):
                if len(raw_output) >= 2:
                    true_score = float(raw_output[self.true_token_id]) if self.true_token_id is not None else 0.0
                    false_score = float(raw_output[self.false_token_id]) if self.false_token_id is not None else 0.0

                    import math
                    exp_true = math.exp(true_score)
                    exp_false = math.exp(false_score)

                    if exp_true + exp_false > 0:
                        return exp_true / (exp_true + exp_false)

            if isinstance(raw_output, torch.Tensor) and raw_output.numel() == 1:
                return raw_output.item()

            return float(raw_output.mean()) if hasattr(raw_output, 'mean') else 0.5

        except Exception as e:
            logger.warning(f"Error processing Qwen3 output: {str(e)}")
            return 0.0


    def print_reranked_documents_and_scores(self, doc_score_pairs: List[tuple]):
        """打印重排序后的文档和分数

        Args:
            doc_score_pairs: 文档和分数的元组列表
        """
        if not doc_score_pairs:
            logger.info("No documents to display")
            return

        logger.info("=" * 80)
        logger.info("Reranked Documents with Scores:")
        logger.info("=" * 80)

        for i, (doc, score) in enumerate(doc_score_pairs, 1):
            doc_preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"[{i}] Score: {score:.4f} | Preview: {doc_preview}...")

        logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reranker_provider = RerankerProvider()


    test_query = "什么是高血压？"
    test_documents = [
        Document(page_content="高血压是一种常见的慢性疾病，特征是动脉血压持续升高。"),
        Document(page_content="糖尿病是一种代谢性疾病，影响人体对血糖的调节能力。"),
        Document(page_content="高血压患者应该注意饮食，减少盐分摄入。"),
        Document(page_content="心脏病是指心脏功能或结构的异常，可能导致心力衰竭。"),
        Document(page_content="适当的运动有助于控制血压水平。"),
    ]

    reranked_docs = reranker_provider.rerank(test_query, test_documents, top_k=3)
    print(f"\nFinal reranked documents: {len(reranked_docs)}")
