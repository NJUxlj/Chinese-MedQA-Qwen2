如何使用 API 目录下的脚本
1. 安装依赖
首先，您需要安装项目的依赖项：

```bash
pip install fastapi uvicorn torch transformers sentence-transformers pydantic psutil scikit-learn rouge
```

2. 设置环境变量
您可以通过环境变量配置API服务的行为：

```bash
# 基本设置
export API_HOST=0.0.0.0
export API_PORT=8000

# 模型相关设置
export DEFAULT_MODEL=qwen2-7b-instruct
export DEFAULT_MODEL_PATH=Qwen/Qwen2-7B-Instruct
export DEFAULT_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
export PRELOAD_MODELS=true

# 知识库相关设置
export KB_INDEX_DIR=knowledge_base/indices
export DEFAULT_KB=medical_kb

# 管理相关设置
export ADMIN_API_KEY=your_secure_api_key_here
```
3. 启动 API 服务
直接启动
从项目根目录运行：

```bash
cd api
python main.py
```
使用 Uvicorn 启动
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
4. API 服务使用示例
API服务启动后，您可以访问以下URL：

API文档: http://localhost:8000/docs
健康检查: http://localhost:8000/health
可用模型列表: http://localhost:8000/api/qa/models


4.1 问答接口使用示例
```python
import requests
import json

# 回答医疗问题
response = requests.post(
    "http://localhost:8000/api/qa/ask",
    json={
        "question": "胃溃疡的症状有哪些？",
        "model_name": None,  # 使用默认模型
        "use_template": True
    }
)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 流式回答
response = requests.post(
    "http://localhost:8000/api/qa/ask_stream",
    json={
        "question": "肝硬化的原因有哪些？",
        "model_name": None,
        "use_template": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        # 解析 SSE 格式数据
        line = line.decode('utf-8')
        if line.startswith('data:'):
            data = json.loads(line[5:])
            if data.get('finished'):
                print("\n完成!")
            else:
                print(data['chunk'], end='', flush=True)
```


4.2 知识检索接口使用示例
```python
import requests
import json

# 检索相关文档
response = requests.post(
    "http://localhost:8000/api/rag/retrieve",
    json={
        "query": "幽门螺杆菌感染如何治疗？",
        "kb_name": "medical_kb",
        "top_k": 5
    }
)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 基于知识库回答问题
response = requests.post(
    "http://localhost:8000/api/rag/ask",
    json={
        "question": "结肠息肉的预防方法有哪些？",
        "kb_name": "medical_kb",
        "model_name": None  # 使用默认模型
    }
)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

4.3 嵌入服务接口使用示例
```python
import requests
import json

# 获取文本嵌入
response = requests.post(
    "http://localhost:8000/api/embedding/embed",
    json={
        "text": "胃溃疡是一种常见的消化系统疾病"
    }
)
print(f"嵌入向量维度: {len(response.json())}")

# 计算文本相似度
response = requests.post(
    "http://localhost:8000/api/embedding/similarity",
    json={
        "text1": "胃溃疡是一种常见的消化系统疾病",
        "text2": "胃溃疡是消化道常见疾病之一"
    }
)
print(f"相似度: {response.json()}")
```


4.4 管理接口使用示例
```python
import requests
import json

# 添加API密钥到请求头
headers = {
    "X-API-Key": "your_secure_api_key_here"
}

# 获取系统状态
response = requests.get(
    "http://localhost:8000/api/admin/status",
    headers=headers
)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 加载模型
response = requests.post(
    "http://localhost:8000/api/admin/models/load?model_name=qwen2-7b-chat&model_path=Qwen/Qwen2-7B-Chat",
    headers=headers
)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

5. Docker 部署
您也可以使用 Docker 部署API服务。创建 Dockerfile：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
构建和运行 Docker 容器：

```

```bash
docker build -t chinese-medqa-api .
docker run -p 8000:8000 -e ADMIN_API_KEY=your_secure_key chinese-medqa-api
```


这套API服务提供了完整的医疗问答功能，包括直接问答、知识检索增强生成、文本嵌入以及系统管理功能。您可以根据需要调整配置和扩展功能。