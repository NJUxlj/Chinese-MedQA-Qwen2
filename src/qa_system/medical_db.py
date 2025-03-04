from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class MedicalVectorDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-med",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
    def create_index(self, docs):
        # 分层索引策略
        chroma_db = Chroma.from_documents(
            docs[:5000], 
            self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        faiss_db = FAISS.from_documents(
            docs[5000:],
            self.embeddings
        )
        
        return HybridRetriever(chroma_db, faiss_db)

class HybridRetriever:
    def __init__(self, chroma, faiss):
        self.chroma = chroma
        self.faiss = faiss

    def similarity_search(self, query, k=5):
        # 混合检索策略
        chroma_results = self.chroma.similarity_search(query, k=k*2)
        faiss_results = self.faiss.similarity_search(query, k=k*2)
        return rerank_and_deduplicate(chroma_results + faiss_results)[:k]