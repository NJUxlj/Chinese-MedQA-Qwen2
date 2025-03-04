from llama_factory import DPOArguments, apply_dpo
from fastllm import load_model
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class MedicalQASystem:
    def __init__(self, model_path, knowledge_base):
        # 加载DPO对齐后的模型
        self.model = load_model(
            model_path,
            quantize=True, 
            device_map="auto",
            flash_attention=True
        )
        
        # 初始化知识库
        self.embeddings = HuggingFaceEmbeddings("BAAI/bge-large-zh")
        self.vector_db = Chroma(
            persist_directory=knowledge_base,
            embedding_function=self.embeddings
        )
    
    def rag_pipeline(self, query):
        # 检索增强
        docs = self.vector_db.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # DPO对齐生成
        prompt = f"""基于以下医疗知识：
{context}
问题：{query}
请给出专业、安全的回答："""
        
        return self.model.generate(prompt)

# 初始化系统
system = MedicalQASystem(
    model_path="models/qwen2-7b-medical",
    knowledge_base="data/knowledge_base/medical_faq"
)