## Chinese-MedQA-Qwen2
本项目是一个基于Qwen2基座模型，以及[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)和[fastllm](https://github.com/ztxz16/fastllm)项目的医疗问答（Medical QA）模型。
该项目的目的，是使用SFT来微调一个使用西医知识来进行疾病诊疗的垂直模型


## 模型介绍
Qwen2 is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. 

Qwen2-7B-Instruct supports a context length of up to 131,072 tokens.


## 数据集

#### SFT数据集


#### DPO数据集



## 权重下载
我们预先加载huggingface上的mradermacher/Med-Qwen2-7B-GGUF的权重到本地
```python
git lfs install
```
```python
git clone https://huggingface.co/mradermacher/Med-Qwen2-7B-GGUF
```

## 微调
一轮LoRA + 一轮DPO，其余步骤根据后续效果再加




## 评估
微调结束后，我们会使用evaluate_model.py来让llama3.1给GPT4o和和Qwen2生成的答案打分。



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

2. 安装cmake
3. 编译
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
6. 
