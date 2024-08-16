# Chinese-MedQA-Qwen2
本项目是一个基于Qwen2基座模型，以及LLaMA-Factory和Fast-LLM项目的医疗问答（Medical QA）模型。
该项目的目的，是使用SFT来微调一个使用西医知识来进行疾病诊疗的垂直模型


# 模型介绍
Qwen2 is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc.

Qwen2-7B-Instruct supports a context length of up to 131,072 tokens.


# 数据集

### SFT数据集


### DPO数据集



# 权重下载
我们预先加载huggingface上的mradermacher/Med-Qwen2-7B-GGUF的权重到本地
```python
git lfs install
```
```python
git clone https://huggingface.co/mradermacher/Med-Qwen2-7B-GGUF
```

# 微调
一轮LoRA + 一轮DPO，其余步骤根据后续效果再加




# 评估
微调结束后，我们会使用evaluate_model.py来让llama3.1给GPT4o和和Qwen2生成的答案打分。



# 如何运行本项目
### 1. 先把项目拉到本地，比如 /你的本地目录/Chinese-MedQA-Qwen2

### 2. 配置LLaMA-Factory
1. 拉取LLaMA-Factory到本地，并确保LLaMA-Factory目录和Chinese-MedQA-Qwen2目录处于同一层级，例如：
   
     -----你的本地目录
                |------Chinese-MedQA-Qwen2
                |------LLaMA-Factory
     ![image](https://github.com/user-attachments/assets/ec6437e1-e2bb-4272-8a33-5f0e8cecfa11)


2. 安装依赖
 
    cd LLaMA-Factory
    pip install -r requirements.txt
    pip install -e ".[torch,metrics]"


    
