# Chinese-MedQA-Qwen2

<!-- Badges -->
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project](https://img.shields.io/badge/Project-MedQA-orange.svg)]()

## 项目简介

本项目是一个基于 **Qwen2 + Agent + RAG** 的医疗问答系统，旨在打通从 SFT/Embedding 训练数据生成，到 SFT 微调、奖励模型微调，再到 DPO/DAPO/GSPO/TRPO 强化学习微调，最终使用 vLLM 部署推理，并通过 AgentFactory 调用医疗多 Agent 会诊系统（MDAgents）进行问诊的完整流水线。

最终目标：基于上述后训练方式，微调出一个使用西医知识进行疾病诊疗的垂直 Qwen2 模型。

## 项目特性

- 🏋️ **多种训练框架**：支持 SFT、DPO、DAPO、GSPO、TRPO、PPO、GRPO 等多种微调算法
- 🔧 **手写 DPOTrainer**：完全自主实现的 DPO 训练器，非第三方库直接调用
- ⚡ **多种推理后端**：支持 vLLM、XInference、Ollama、Transformers 多种推理方式
- 🔍 **混合检索系统**：集成相似度、BM25、L2距离、KNN 等多种检索算法
- 🤖 **Multi-Agent 会诊**：基于 MDAgents 的多专科医疗会诊系统
- 📚 **知识库增强**：支持 FAISS 向量库 + Milvus + Neo4j 知识图谱

## 技术栈总结

| 层次 | 技术 |
|------|------|
| **基础模型** | Qwen2 / Qwen3（支持本地部署和 API 调用） |
| **SFT 微调** | HuggingFace Trainer + Accelerate |
| **DPO 微调** | 手写 DPOTrainer（参考 LLaMA-Factory） |
| **强化学习** | VeRL 框架（支持 PPO/GRPO/DAPO/SPPO/SPiNO/TRPO） |
| **推理加速** | vLLM、XInference、Ollama、Transformers |
| **向量数据库** | Milvus + FAISS |
| **图数据库** | Neo4j |
| **RAG 框架** | LangChain (langchain-core, langchain-community) |
| **Agent 框架** | LangChain + 自定义 Multi-Agent (MDAgents) |
| **配置管理** | Hydra + OmegaConf |
| **Web 框架** | FastAPI + Gradio |
| **评估指标** | BLEU、ROUGE、Math（自定义） |

## 项目架构图

```
Chinese-MedQA-Qwen2
├── src/                          # 核心源码
│   ├── agent/                    # Agent 模块
│   │   ├── base_agent.py         # 基础 Agent 类
│   │   ├── medical_agent.py      # 医疗 Agent
│   │   ├── agent_factory.py      # Agent 工厂
│   │   ├── mdagents/             # 多专科会诊系统
│   │   │   ├── core/             # 核心控制器
│   │   │   └── agents/           # 各专科 Agent
│   │   └── tools/                # Agent 工具集
│   │
│   ├── rag/                      # RAG 流水线
│   │   ├── rag_pipeline.py       # RAG 主流程
│   │   ├── query_processor.py   # 查询处理
│   │   ├── context_builder.py    # 上下文构建
│   │   └── response_generator.py # 响应生成
│   │
│   ├── knowledge_base/            # 知识库
│   │   ├── retrieval/            # 检索器实现
│   │   │   ├── similarity_retriever.py  # 相似度检索
│   │   │   ├── bm25_retriever.py        # BM25 检索
│   │   │   ├── l2_retriever.py          # L2 距离检索
│   │   │   └── knn_retriever.py         # KNN 检索
│   │   ├── embedding/            # 嵌入管理
│   │   ├── kg/                    # 知识图谱
│   │   └── milvus/                # Milvus 接口
│   │
│   ├── training/                  # 训练模块
│   │   ├── trainer/               # 多种训练器
│   │   │   ├── sft_trainer.py     # SFT 训练器
│   │   │   ├── dpo_trainer.py     # DPO 训练器（手写）
│   │   │   ├── dapo_trainer.py     # DAPO 训练器
│   │   │   ├── trpo_trainer.py     # TRPO 训练器
│   │   │   └── reward_model_trainer.py  # 奖励模型
│   │   ├── dataset/               # 数据集处理
│   │   │   └── medical_dataset.py # 医疗数据集类
│   │   └── run_trainer/           # 训练启动脚本
│   │       ├── run_sft_trainer.py
│   │       └── run_dpo_trainer.py
│   │
│   ├── inference/                  # 推理模块
│   │   ├── vllm_inference.py       # vLLM 推理
│   │   ├── xinference_inference.py # XInference 推理
│   │   ├── ollama_inference.py     # Ollama 推理
│   │   └── transformers_inference.py # Transformers 推理
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── base_evaluator.py       # 评估器基类
│   │   ├── bleu_rouge_evaluator.py # BLEU/ROUGE 评估
│   │   ├── medqa_llm_evaluator.py   # MedQA LLM 评估
│   │   └── math_evaluator.py        # 数学题评估
│   │
│   ├── models/                     # 模型封装
│   │   ├── base_model.py           # 基础模型类
│   │   ├── qwen_model.py           # Qwen 模型封装
│   │   └── api_model.py            # API 模型封装
│   │
│   ├── api/                        # FastAPI 服务
│   │   ├── main.py                 # API 主入口
│   │   └── routers/                # API 路由
│   │
│   ├── config/                     # 配置模块
│   │   ├── config.yaml             # 统一配置文件
│   │   ├── settings.py             # 配置加载器
│   │   └── deepspeed_config/       # DeepSpeed 配置
│   │
│   └── utils/                      # 工具模块
│       ├── logger.py               # 日志管理
│       ├── metrics.py              # 评估指标
│       └── text_utils.py           # 文本处理
│
├── examples/                       # 示例代码
│   ├── train/                      # 训练 YAML 配置
│   │   ├── sft.yaml
│   │   ├── dpo.yaml
│   │   ├── grpo.yaml
│   │   ├── ppo.yaml
│   │   └── reward_model.yaml
│   └── mdagents_usage_examples.py  # MDAgents 使用示例
│
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后数据
│   └── indices/                    # FAISS 索引
│
├── requirements.txt                # 项目依赖
└── README.md
```

## 数据集

### SFT 数据集
- 来源: [`ticoAg/Chinese-medical-dialogue`](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue)
- 字段格式: `{"instruction", "input", "output"}`

### DPO 数据集
- 来源: [`Morefreedai/medical-dpo-v1`](https://huggingface.co/datasets/Morefreedai/medical-dpo-v1)
- 字段格式: `{"prompt", "chosen", "rejected"}`

### RL 训练数据
- GSM8K 数学题: `~/data/rlhf/gsm8k/`

## 模型权重

主要基座模型: **`Qwen/Qwen3-4B`** (HuggingFace)

模型路径通过环境变量配置:
- `LOCAL_MODEL_PATH` - 本地模型路径
- `EMBEDDING_MODEL_PATH` - Embedding 模型路径
- `RERANKER_MODEL_PATH` - Reranker 模型路径

## 如何运行本项目

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export LOCAL_MODEL_PATH="/path/to/Qwen3-4B"
export EMBEDDING_MODEL_PATH="/path/to/embedding/model"
export RERANKER_MODEL_PATH="/path/to/reranker/model"
```

### 3. 启动 API 服务

```bash
docker compose up
cd src/api
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 运行 SFT 训练

```bash
python src/training/run_trainer/run_sft_trainer.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --train_data_dir ./data \
    --output_dir ./output/sft \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --finetuning_type lora \
    --lora_rank 16
```

### 5. 运行 DPO 训练

```bash
python src/training/run_trainer/run_dpo_trainer.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --train_data_dir ./data \
    --output_dir ./output/dpo \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_epochs 3
```

### 6. 使用 YAML 配置训练

```bash
# SFT 训练
python src/training/run_trainer/run_sft_trainer.py --config examples/train/sft.yaml

# DPO 训练
python src/training/run_trainer/run_dpo_trainer.py --config examples/train/dpo.yaml
```

### 7. vLLM 部署

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000
```

## Evaluation 评估

项目提供多种评估器，位于 `src/evaluation/`:

| 评估器 | 说明 |
|--------|------|
| `BleuRougeEvaluator` | BLEU/ROUGE 指标评估 |
| `MedQALLMEvaluator` | 基于 LLM 的医疗 QA 评估 |
| `MathEvaluator` | 数学问题评估 |
| `CodeEvaluator` | 代码执行评估 |

评估配置在 `src/config/config.yaml` 的 `evaluator` 部分。

## 参考项目

- **Agent**: [AgentGPT](https://github.com/reworkd/AgentGPT.git), [CAMEL](https://github.com/camel-ai/camel.git)
- **Medical RAG**: [Medical-Graph-RAG](https://github.com/SuperMedIntel/Medical-Graph-RAG.git)
- **RAG 架构**: [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat.git)
- **DPO 实现参考**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## 修复日志

- **2026-04-14**: 修复 17 项严重问题（P0），包括导入路径修正、配置安全加固、训练逻辑修复、LoggerManager 统一、训练 YAML 配置创建。详见 [docs/修复文档/FIX_1.md](docs/修复文档/FIX_1.md)

## License

MIT License
