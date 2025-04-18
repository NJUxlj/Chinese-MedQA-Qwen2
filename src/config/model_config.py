# 选择模型类型  
use_api = False  # 是否使用API（智谱API）  
use_fastllm = False  # 是否使用FastLLM加速  
use_vllm = False  # 是否使用VLLM加速  

# 本地模型配置  
model_path = "path/to/qwen2/model"  # 本地模型路径  
device = "cuda:0"  # 设备  

# API配置（如果使用API）  
api_key = "your-api-key"  
api_base = "https://api.example.com"  
model_name = "Qwen2-7B-Chat"  

# 知识库配置  
retriever_type = "knn"  # 检索类型：knn, similarity, bm25, l2  
embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"  
index_path = "data/indices/faiss_index.bin"  
top_k = 5  # 默认检索数量  

# 推理配置  
max_new_tokens = 1024  
temperature = 0.7  
top_p = 0.9  

# 服务配置  
host = "0.0.0.0"  
port = 8000  
debug = False  
verbose = False  