## Chinese-MedQA-Qwen2
- 本项目是一个基于Qwen2+Agent+RAG的医疗问答系统
- 该项目的目的，是使用`SFT+DPO`来微调一个使用西医知识来进行疾病诊疗的垂直qwen2模型, 并将SFT+DPO微调后的模型(也可以用智谱api模型调用进行替换)的回答文本和本地知识库中的文本做匹配，然后使用RAG的方式(参考`Longchain-chatchat`项目)来将原始回答和匹配的top-k个文本段进行拼接，然后再进行回答。


## 项目内容
1. 手动构建 SFT+DPO 的Trainer.(SFT由huggingface的Trainer实现，DPO 是由 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 实现, 用户也可以选择我们手动编写的DPOTrainer【注意，这是我手写的！和trl库里的那个DPOTrainer不是同一个】)
2. 推理实现：用户可以选择两种推理方式：1.使用 [fastllm](https://github.com/ztxz16/fastllm)（一个基于C++的推理库）加速推理。2.用户也可以切换成使用 VLLM 进行推理加速
3. 本项目也参考了LongChain-Chatchat的项目框架：【1】包括Ollama，XInference的基本使用(主要是模型的加载与推理)。
4. 参考了LongChain-Chatchat的架构，在Chinese-MedQA的文档匹配算法中加入KNN（原本只有相似度、BM25、L2_distance）。【3】使用FAISS构建了本地医疗知识库。


## 参考的项目
- Agent部分 参考了：
  1.  [AgentGPT](https://github.com/reworkd/AgentGPT.git)
  2.  [Camel](https://github.com/camel-ai/camel.git)
- 医疗RAG实现+工具调用+数据库部分参考了 [Medical-Graph-RAG](https://github.com/SuperMedIntel/Medical-Graph-RAG.git)
- RAG检索算法+项目结构参考了 [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat.git)



## 技术栈总结
1. 基础模型: Qwen2（同时支持本地部署和智谱API调用）
2. 微调框架:
  - SFT: 基于Hugging Face Trainer实现
  - DPO: 手动构建的DPOTrainer，参考LLaMA-Factory
3. 推理加速:
  - FastLLM（基于C++的推理库）
  - VLLM（大规模部署时的推理加速）
4. 知识库与检索:
  - FAISS向量数据库（高效相似性搜索）
  - 多种检索算法：相似度、BM25、L2距离、KNN
5. Agent实现:
  - 参考AgentGPT和Camel项目
6. 项目框架:
  - 参考Langchain-Chatchat和Medical-Graph-RAG的项目结构


## 模型介绍
Qwen2 is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. 

Qwen2-7B-Instruct supports a context length of up to 131,072 tokens.


## 数据集

#### SFT数据集
- 字段格式：{"instruction"..., "input":..., "output":...}
```python
from datasets import load_dataset

ds = load_dataset("ticoAg/Chinese-medical-dialogue")
```

#### DPO数据集
```python

from datasets import load_dataset

ds = load_dataset("Morefreedai/medical-dpo-v1")

```


## 权重下载


## SFT
一轮LoRA/SFT + 一轮DPO，其余步骤根据后续效果再加




## Evaluation
微调结束后，我们会使用evaluate_model.py来让llama3.1给GPT4o和和Qwen2生成的答案打分。


## Environment Config
- AutoDL Cloud Platform
  
![env](image/env.png)

- then, make sure to pre-download the model weight (e.g. Qwen2.5-1.5B on the huggingface) to the local storage (e.g., `/root/autodl-tmp/models/Qwen2.5-1.5B`).




## 如何运行本项目
#### 1. 先把项目拉到本地，比如 /你的本地目录/Chinese-MedQA-Qwen2

#### 2. 配置LLaMA-Factory
1. 拉取LLaMA-Factory到本地，并确保LLaMA-Factory目录和Chinese-MedQA-Qwen2目录处于同一层级，例如：
 ```python  
     -----你的本地目录
                |------Chinese-MedQA-Qwen2
                |------LLaMA-Factory
```


2. 安装依赖
 ```python  
    cd LLaMA-Factory
    pip install -r requirements.txt
    pip install -e ".[torch,metrics]"
```


#### 3. 配置fastllm
```python
git clone https://github.com/ztxz16/fastllm.git

cd ./fastllm    
```

```python
pip install -r requirements.txt
```

```python  
-----你的本地目录
          |------Chinese-MedQA-Qwen2
          |------LLaMA-Factory
          |------fastllm
```
1. 安装gcc
```python
# 确认是否已经安装
gcc --version

# 安装
# For Debian based distributions like Ubuntu
sudo apt-get install gcc

# For RPM-based distributions like CentOS
sudo yum install gcc
```

3. 安装cmake
```python
sudo apt install cmake -y
cmake  --version
```

4. 编译
 ```python
bash install.sh -DUSE_CUDA=ON # 编译GPU版本
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 # 可以指定CUDA架构，如4090使用89架构, A100使用80架构
```
4. 跑起来
```python
# openai api server
# 需要安装依赖: pip install -r requirements-server.txt
# 这里在8080端口打开了一个模型名为qwen的server
python3 -m ftllm.server -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080 --model_name qwen

# 使用float16精度的模型对话
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ 

# 在线量化为int8模型对话
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ --dtype int8

# webui
# 需要安装依赖: pip install streamlit-chat
python3 -m ftllm.webui -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080
```




