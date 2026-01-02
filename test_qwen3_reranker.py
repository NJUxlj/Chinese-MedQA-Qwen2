#!/usr/bin/env python3
"""
测试 Qwen3-Reranker-0.6B 模型的脚本

这个脚本演示了如何正确使用 Qwen3-Reranker 模型进行文档重排序。
Qwen3-Reranker 使用特殊的聊天模板和输出机制。
"""
import os
import sys
import torch
import logging

from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3RerankerTester:
    """Qwen3-Reranker 模型测试器
    
    专门用于测试 Qwen3-Reranker 模型的正确使用方式：
    - 使用聊天模板格式化输入
    - 从 "yes"/"no" 的概率计算相关性分数
    """
    
    def __init__(self, model_path: str = "code/models/Qwen3-Reranker-0.6B", device: str = "cpu"):
        """初始化测试器
        
        Args:
            model_path: 模型路径
            device: 设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.true_token_id = None
        self.false_token_id = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        logger.info(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        logger.info(f"Loading model from: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        # 获取 yes/no 的 token ID
        self.true_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        logger.info(f"Model loaded successfully. Yes token: {self.true_token_id}, No token: {self.false_token_id}")
    
    def _format_instruction(self, instruction: str, query: str, doc: str):
        """格式化输入为聊天模板
        
        Args:
            instruction: 任务指令
            query: 查询
            doc: 文档
            
        Returns:
            格式化的消息列表
        """
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]
    
    def _compute_score(self, logits: torch.Tensor) -> float:
        """计算相关性分数
        
        Args:
            logits: 模型的 logits 输出
            
        Returns:
            相关性分数 (0-1)
        """
        # 获取最后一个 token 的 logits
        last_token_logits = logits[0, -1, :]
        
        # 获取 yes 和 no 的 logit
        true_logit = last_token_logits[self.true_token_id].item()
        false_logit = last_token_logits[self.false_token_id].item()
        
        # 使用 softmax 计算概率
        true_score = torch.exp(torch.tensor(true_logit)).item()
        false_score = torch.exp(torch.tensor(false_logit)).item()
        
        # 计算相关性分数
        score = true_score / (true_score + false_score)
        return score
    
    def rerank(self, query: str, documents: list, instruction: str = None, top_k: int = 5):
        """对文档进行重排序
        
        Args:
            query: 查询字符串
            documents: 文档列表 (字符串)
            instruction: 任务指令
            top_k: 返回的 top_k 个文档
            
        Returns:
            重排序后的文档列表 (包含相关性分数)
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        if not query:
            logger.warning("Query is empty, returning original documents")
            return documents[:top_k]
        
        if not documents:
            logger.warning("Documents list is empty")
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        
        # 格式化所有输入
        formatted_inputs = []
        for doc in documents:
            messages = self._format_instruction(instruction, query, doc)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            formatted_inputs.append(text)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 计算每个文档的分数
        scores = []
        for i in range(len(documents)):
            # 对于 padding 的文档，跳过
            if inputs["attention_mask"][i].sum() == 0:
                scores.append(0.0)
                continue
            
            # 获取有效长度的 logits
            valid_length = inputs["attention_mask"][i].sum().item()
            doc_logits = logits[i : i + 1, :valid_length, :]
            score = self._compute_score(doc_logits)
            scores.append(score)
        
        # 排序
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranking completed. Top {top_k} documents selected.")
        
        # 打印结果
        self._print_results(query, doc_score_pairs[:top_k])
        
        return doc_score_pairs[:top_k]
    
    def _print_results(self, query: str, results: list):
        """打印重排序结果
        
        Args:
            query: 查询
            results: 重排序结果
        """
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info("=" * 80)
        logger.info("Reranked Documents with Scores:")
        logger.info("-" * 80)
        
        for i, (doc, score) in enumerate(results, 1):
            doc_preview = doc[:100].replace('\n', ' ')
            logger.info(f"[{i}] Score: {score:.4f} | Preview: {doc_preview}...")
        
        logger.info("=" * 80)


def test_medical_qa():
    """测试医学问答场景"""
    logger.info("\n" + "=" * 80)
    logger.info("测试场景 1: 医学问答 - 什么是高血压？")
    logger.info("=" * 80)
    
    query = "什么是高血压？"
    documents = [
        "高血压是一种常见的慢性疾病，特征是动脉血压持续升高。",
        "糖尿病是一种代谢性疾病，影响人体对血糖的调节能力。",
        "高血压患者应该注意饮食，减少盐分摄入。",
        "心脏病是指心脏功能或结构的异常，可能导致心力衰竭。",
        "适当的运动有助于控制血压水平。",
        "感冒是由病毒引起的上呼吸道感染，常见症状包括咳嗽、流鼻涕和发热。",
    ]
    
    reranker = Qwen3RerankerTester(
        model_path="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Reranker-0.6B",
        device="cpu"
    )
    
    results = reranker.rerank(query, documents, top_k=3)
    
    return results


def test_symptom_matching():
    """测试症状匹配场景"""
    logger.info("\n" + "=" * 80)
    logger.info("测试场景 2: 症状匹配 - 头痛应该挂什么科？")
    logger.info("=" * 80)
    
    query = "头痛应该挂什么科？"
    documents = [
        "神经内科专门治疗头痛、癫痫、中风等神经系统疾病。",
        "心血管内科负责治疗高血压、冠心病、心脏病等循环系统疾病。",
        "骨科主要治疗骨折、关节炎、颈椎病等骨骼肌肉系统疾病。",
        "眼科负责治疗近视、白内障、青光眼等眼部疾病。",
        "如果头痛伴有视力模糊，建议先去眼科检查排除眼部问题。",
    ]
    
    reranker = Qwen3RerankerTester(
        model_path="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Reranker-0.6B",
        device="cpu"
    )
    
    results = reranker.rerank(query, documents, top_k=3)
    
    return results


def test_drug_information():
    """测试药物信息查询场景"""
    logger.info("\n" + "=" * 80)
    logger.info("测试场景 3: 药物信息 - 阿司匹林的作用和副作用")
    logger.info("=" * 80)
    
    query = "阿司匹林的作用和副作用"
    documents = [
        "阿司匹林是一种常用的解热镇痛药，主要用于治疗发热、头痛、关节痛等症状。",
        "高血压患者应在医生指导下使用降压药物，定期监测血压。",
        "阿司匹林可以抑制血小板聚集，预防血栓形成，但可能引起胃肠道不适。",
        "糖尿病患者需要注意饮食控制，避免高糖食物。",
        "长期使用阿司匹林可能增加出血风险，应在医生指导下使用。",
    ]
    
    reranker = Qwen3RerankerTester(
        model_path="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Reranker-0.6B",
        device="cpu"
    )
    
    results = reranker.rerank(query, documents, top_k=3)
    
    return results


def test_english_documents():
    """测试英文文档场景"""
    logger.info("\n" + "=" * 80)
    logger.info("测试场景 4: 英文文档 - What is machine learning?")
    logger.info("=" * 80)
    
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Deep learning is a technique used in machine learning that models high-level abstractions in data.",
        "Python is a popular programming language for machine learning and data science.",
        "Neural networks are computing systems inspired by biological neural networks in the human brain.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
    ]
    
    reranker = Qwen3RerankerTester(
        model_path="/Users/xiniuyiliao/Desktop/code/models/Qwen3-Reranker-0.6B",
        device="cpu"
    )
    
    results = reranker.rerank(query, documents, top_k=3)
    
    return results


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Qwen3-Reranker-0.6B 模型测试")
    logger.info("=" * 80)
    
    # 测试不同的场景
    test_medical_qa()
    test_symptom_matching()
    test_drug_information()
    test_english_documents()
    
    logger.info("\n" + "=" * 80)
    logger.info("所有测试完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
