# 医疗AI Agent

一个基于Qwen2或ZhipuAI的医疗AI Agent，具备对PubMed和本地医疗知识库的搜索、匹配、整合、反思和纠错能力。

## 功能特点

- **多源信息检索**: 同时支持PubMed在线搜索和本地医疗知识库检索
- **智能信息整合**: 将多个来源的信息进行整合，提供全面的答案
- **自反思与纠错**: 通过反思机制提高回答质量，纠正可能的错误
- **灵活模型选择**: 支持Qwen2和ZhipuAI两种大型语言模型
- **向量化本地知识**: 支持PDF、TXT、CSV等格式的本地医疗文档索引和检索
- **用户友好界面**: 基于Gradio的简洁Web界面

## 系统架构


1. **大语言模型(LLM)层**: 使用Qwen2或ZhipuAI作为核心推理引擎
2. **检索增强生成(RAG)层**: 负责搜索、匹配和整合信息
3. **知识源接入层**: 连接PubMed和本地医疗知识库
4. **反思与自校正层**: 提供输出校验和改进
5. **API接口层**: 提供服务接口



## 项目结构
```Plain Text
medical_ai_agent/  
├── config/  
│   └── config.yaml  
├── data/  
│   └── local_knowledge/  
├── src/  
│   ├── agent/  
│   │   ├── __init__.py  
│   │   ├── agent.py  
│   │   └── reflection.py  
│   ├── llm/  
│   │   ├── __init__.py  
│   │   ├── qwen_model.py  
│   │   └── zhipu_model.py  
│   ├── retrieval/  
│   │   ├── __init__.py  
│   │   ├── knowledge_base.py  
│   │   ├── pubmed.py  
│   │   └── vector_store.py  
│   └── utils/  
│       ├── __init__.py  
│       └── helpers.py  
├── app.py  
├── requirements.txt  
└── README.md  
```

## 安装与配置

### 1. 环境需求

- Python 3.9+
- 足够的GPU内存(使用本地模型时)或API密钥(使用云API时)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置项目

编辑`config/config.yaml`文件，设置以下配置：

- LLM提供商选择(Qwen2或ZhipuAI)及API密钥
- PubMed API设置
- 本地知识库路径
- 向量存储设置

### 4. 准备本地知识库

将医疗PDF、文本文件、CSV等文件放入`data/local_knowledge/`目录中。

## 使用方法

### 1. 启动应用

```bash
python app.py --config ./config/config.yaml
```

### 2. 索引本地知识

首次使用时，需要在Web界面中点击"索引本地文档"按钮，将本地医疗文档加入向量数据库。

### 3. 提问医疗问题

在输入框中输入医疗问题，系统会自动：
- 判断是否需要检索外部信息
- 执行必要的PubMed搜索或本地知识库检索
- 整合信息并生成回答
- 对回答进行反思和可能的纠正
- 返回最终答案

## 示例问题

以下是一些可以尝试的示例问题：

- "Omicron变种的主要症状有哪些？"
- "2型糖尿病的最新治疗指南是什么？"
- "HER2阳性乳腺癌的靶向治疗有哪些进展？"
- "长新冠综合征的病理机制研究进展如何？"
- "CAR-T细胞疗法在血液系统恶性肿瘤中的应用状况？"

## 项目结构

```
medical_ai_agent/
├── config/
│   └── config.yaml           # 配置文件
├── data/
│   └── local_knowledge/      # 本地知识文档目录
├── src/
│   ├── agent/                # Agent核心实现
│   ├── llm/                  # 大语言模型接口
│   ├── retrieval/            # 检索系统实现
│   └── utils/                # 工具函数
├── app.py                    # 主应用程序
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

## 注意事项

- 本系统提供的医疗信息仅供参考，不构成医疗建议，重要医疗决策请咨询专业医生
- 使用PubMed API时请遵守NCBI的使用政策和限制
- 使用Qwen2和ZhipuAI API时请遵循相应的使用条款



## 总结
这是一个基于Qwen2或ZhipuAI的医疗AI Agent系统，该系统具有以下核心功能：

灵活的LLM选择: 支持Qwen2和ZhipuAI两种大型语言模型，可通过配置切换
全面的检索能力: 整合PubMed在线检索和本地医疗知识库检索
智能的工具调用: 通过函数调用机制自动选择和使用合适的工具
高级反思机制: 使用Chain-of-Thought或`ReAct`策略对初始答案进行反思和改进
用户友好界面: 基于Gradio的Web界面，支持实时交互和知识库管理
系统采用了模块化的设计，主要组件包括：

LLM接口层 (src/llm/)
检索系统 (src/retrieval/)
Agent核心逻辑 (src/agent/)
辅助工具 (src/utils/)
该项目充分利用了现代深度学习框架，包括：

transformers: 用于大语言模型接口
sentence-transformers: 用于文本嵌入
langchain: 用于文档处理和向量存储
gradio: 用于Web界面
系统还具备良好的可扩展性：可以通过配置文件轻松添加新的知识源、调整反思策略或切换不同的LLM提供商。

使用此系统，用户可以提出复杂的医疗问题，系统将自动从PubMed和本地知识库中获取相关信息，通过大模型进行整合和分析，并经过反思和纠错后提供准确的回答。



## 许可证

本项目采用MIT许可证
