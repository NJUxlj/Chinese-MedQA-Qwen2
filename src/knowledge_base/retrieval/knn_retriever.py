# knowledge_base/retrieval/knn_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os,sys  
import numpy as np  
import logging  
import pickle  
from pathlib import Path  
import json  
import time  
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import faiss  
except ImportError:
    raise ImportError("FAISS not installed. Please install it with 'pip install faiss-cpu', if you are using macbook, please use 'brew install swig && pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple' or 'conda install -c conda-forge faiss-cpu'  or  'conda install -c pytorch faiss-cpu' instead.")


from langchain_core.documents import Document  

from knowledge_base.retrieval.base_retriever import BaseRetriever  
from providers.embedding_provider import EmbeddingProvider  
from config.settings import settings

logger = logging.getLogger(__name__)  

class KNNRetriever(BaseRetriever):  
    """  
    Retriever that uses KNN with FAISS for efficient similarity search.  
    FAISS provides fast and scalable similarity search.  
    
    Best Practices:  
    - Use Flat index for small datasets (< 10K) requiring exact search  
    - Use IVF index for medium datasets with approximate search  
    - Use HNSW index for large-scale, high-speed search scenarios  
    - Inner Product with normalized vectors = Cosine Similarity  
    """  
    
    def __init__(
        self,
        config=None,
        embedding_manager: EmbeddingProvider=None,
    ):
        """
        Initialize the KNN retriever.

        Args:
            config: KNNRetrieverConfig with index parameters
            embedding_manager: Embedding manager for document and query embedding
        """
        if config is None:
            config = settings.retriever.knn
        if embedding_manager is None:
            embedding_manager = EmbeddingProvider()
        super().__init__(config=config)

        self.embedding_manager = embedding_manager
        self.score_threshold = self.config.score_threshold if hasattr(self.config, 'score_threshold') else 0.1
        self.index_type = self.config.index_type.upper() if hasattr(self.config, 'index_type') else 'FLAT'
        self.n_list = self.config.n_list if hasattr(self.config, 'n_list') else 100
        self.m = self.config.m if hasattr(self.config, 'm') else 16  
        
        self.documents: List[Document] = []  
        self.document_ids: List[str] = []  
        self.index = None  
        self.dimension = None  
        self._index_stats = {}  
    
    def _create_index(self, dimension: int) -> faiss.Index:  
        """  
        Create a FAISS index with the specified parameters.  
        
        Args:  
            dimension: Dimensionality of the vectors  
            
        Returns:  
            FAISS index  
        """  
        metric = faiss.METRIC_INNER_PRODUCT  
        
        if self.index_type == "FLAT":  
            return faiss.IndexFlatIP(dimension)  
            
        elif self.index_type == "IVF":  
            n_list = min(self.n_list, 1000)  
            quantizer = faiss.IndexFlatIP(dimension)  
            index = faiss.IndexIVFFlat(quantizer, dimension, n_list, metric)  
            index.nprobe = max(1, min(n_list // 10, 50))  
            return index  
            
        elif self.index_type == "HNSW":  
            m = min(self.m, min(dimension // 2, 64))  
            index = faiss.IndexHNSWFlat(dimension, m, metric)  
            index.hnsw.efConstruction = min(200, 2 * self.n_list)  
            index.hnsw.efSearch = min(128, max(16, self.n_list))  
            return index  
            
        elif self.index_type == "PQ":  
            m = min(8, dimension // 2)  
            return faiss.IndexPQ(dimension, m, 8)  
            
        else:  
            raise ValueError(f"Unknown index type: {self.index_type}. Supported types: FLAT, IVF, HNSW, PQ")  
    
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        if not documents:  
            return  
            
        logger.info(f"Adding {len(documents)} documents to KNN retriever")  
        start_time = time.time()  
        
        doc_ids = [doc.metadata.get("doc_id", str(len(self.documents) + i)) for i, doc in enumerate(documents)]  
        
        try:  
            embeddings_dict = self.embedding_manager.embed_documents(documents)  
            
            vectors = []  
            valid_docs = []  
            valid_ids = []  
            
            for i, doc in enumerate(documents):  
                doc_id = doc_ids[i]  
                if doc_id in embeddings_dict:  
                    vector = np.array(embeddings_dict[doc_id]).astype('float32')  
                    faiss.normalize_L2(vector.reshape(1, -1))  
                    vectors.append(vector)  
                    valid_docs.append(doc)  
                    valid_ids.append(str(doc_id))  
                    
            if not vectors:  
                logger.warning("No valid embeddings to add to KNN retriever")  
                return  
                
            vectors_array = np.vstack(vectors)  
            self._index_stats['last_add_count'] = len(vectors)  
            
            if self.index is None:  
                self.dimension = vectors_array.shape[1]  
                self.index = self._create_index(self.dimension)  
                logger.info(f"Created new {self.index_type} index with dimension {self.dimension}")  
                
                if self.index_type == "IVF":  
                    logger.info(f"Training IVF index with {vectors_array.shape[0]} vectors")  
                    self.index.train(vectors_array)  
            
            self.index.add(vectors_array)  
            self.documents.extend(valid_docs)  
            self.document_ids.extend(valid_ids)  
            
            self._index_stats['total_vectors'] = self.index.ntotal  
            logger.info(f"Added {len(valid_docs)} documents in {time.time()-start_time:.2f}s (total: {self._index_stats['total_vectors']})")  
            
        except Exception as e:  
            logger.error(f"Error adding documents to KNN retriever: {str(e)}")  
            raise  
    
    def delete_documents(self, document_ids: List[str]) -> None:  
        """  
        Delete documents from the retriever's index.  
        
        Args:  
            document_ids: List of document IDs to delete  
        """  
        if not document_ids or not self.documents:  
            logger.warning("No documents to delete or empty retriever")  
            return  
            
        # FAISS doesn't support direct removal, so we need to rebuild the index  
        
        # Find indices of documents to keep  
        indices_to_keep = []  
        docs_to_keep = []  
        ids_to_keep = []  
        
        for i, doc_id in enumerate(self.document_ids):  
            if doc_id not in document_ids:  
                indices_to_keep.append(i)  
                docs_to_keep.append(self.documents[i])  
                ids_to_keep.append(doc_id)  
                
        if not indices_to_keep:  
            # All documents are deleted  
            self.documents = []  
            self.document_ids = []  
            self.index = None  
            logger.info("All documents deleted from KNN retriever")  
            return  
            
        # Re-embed documents to keep  
        try:  
            embeddings_dict = self.embedding_manager.embed_documents(docs_to_keep)  
            
            # Create vectors array for FAISS  
            vectors = []  
            
            for doc_id in ids_to_keep:  
                if doc_id in embeddings_dict:  
                    vector = np.array(embeddings_dict[doc_id]).astype('float32')  
                    
                    # Normalize vector for inner product  
                    faiss.normalize_L2(vector.reshape(1, -1))  
                    
                    vectors.append(vector)  
                    
            # Convert to numpy array  
            vectors_array = np.vstack(vectors)  
            
            # Create new index  
            self.index = self._create_index(self.dimension)  
            
            # Train if needed  
            if self.index_type == "IVF":  
                self.index.train(vectors_array)  
            
            # Add vectors to index  
            self.index.add(vectors_array)  
            
            # Update storage  
            self.documents = docs_to_keep  
            self.document_ids = ids_to_keep  
            
            logger.info(f"Rebuilt KNN retriever after deletion with {len(self.documents)} documents")  
            
        except Exception as e:  
            logger.error(f"Error rebuilding KNN retriever: {str(e)}")  
            raise  
    
    def search(  
        self,   
        query: str,   
        top_k: int = 5,   
        score_threshold: Optional[float] = None,  
        ef_search: Optional[int] = None  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search for documents similar to the query.  
        
        Args:  
            query: Query string  
            top_k: Number of top results to return  
            score_threshold: Minimum similarity score threshold (overrides instance threshold)  
            ef_search: HNSW search parameter (overrides default)  
            
        Returns:  
            List of (document, score) tuples  
        """  
        if not self.documents or self.index is None:  
            logger.warning("No documents in KNN retriever or index not initialized")  
            return []  
            
        if score_threshold is None:  
            score_threshold = self.score_threshold  
            
        try:  
            query_embedding = np.array(self.embedding_manager.embed_query(query)).astype('float32')  
            faiss.normalize_L2(query_embedding.reshape(1, -1))  
            
            search_k = min(top_k * 2, len(self.documents))  
            
            if self.index_type == "HNSW" and ef_search:  
                self.index.hnsw.efSearch = ef_search  
            
            scores, indices = self.index.search(query_embedding.reshape(1, -1), search_k)  
            
            results = []  
            for i, idx in enumerate(indices[0]):  
                if idx != -1 and i < top_k:  
                    score = float(scores[0][i])  
                    if score >= score_threshold:  
                        results.append((self.documents[idx], score))  
                        
            self._index_stats['last_search_time'] = time.time()  
            return results  
            
        except Exception as e:  
            logger.error(f"Error searching in KNN retriever: {str(e)}")  
            return []  
    
    def save(self, directory: str) -> None:  
        """  
        Save the retriever to a directory.  
        
        Args:  
            directory: Directory to save to  
        """  
        save_dir = Path(directory)  
        save_dir.mkdir(parents=True, exist_ok=True)  
        
        # Save documents  
        with open(save_dir / "documents.pkl", "wb") as f:  
            pickle.dump(self.documents, f)  
            
        # Save document IDs  
        with open(save_dir / "document_ids.json", "w") as f:  
            json.dump(self.document_ids, f)  
            
        # Save FAISS index  
        if self.index is not None:  
            faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))  
            
        # Save config  
        config = {  
            "name": self.config.name,  
            "score_threshold": self.score_threshold,  
            "embedding_model_name": self.embedding_manager.embedding_model_name,  
            "index_type": self.index_type,  
            "n_list": self.n_list,  
            "m": self.m,  
            "dimension": self.dimension  
        }  
        with open(save_dir / "config.json", "w") as f:  
            json.dump(config, f)  
            
        logger.info(f"Saved KNN retriever to {directory}")  
    
    def load(self, directory: str) -> None:  
        """  
        Load the retriever from a directory.  
        
        Args:  
            directory: Directory to load from  
        """  
        load_dir = Path(directory)  
        
        # Check if directory exists  
        if not load_dir.exists():  
            raise FileNotFoundError(f"Directory {directory} not found")  
            
        # Load documents  
        with open(load_dir / "documents.pkl", "rb") as f:  
            self.documents = pickle.load(f)  
            
        # Load document IDs  
        with open(load_dir / "document_ids.json", "r") as f:  
            self.document_ids = json.load(f)  
            
        # Load config  
        with open(load_dir / "config.json", "r") as f:  
            config = json.load(f)  
            self.name = config.get("name", self.name)  
            self.score_threshold = config.get("score_threshold", self.score_threshold)  
            self.index_type = config.get("index_type", self.index_type)  
            self.n_list = config.get("n_list", self.n_list)  
            self.m = config.get("m", self.m)  
            self.dimension = config.get("dimension", self.dimension)  
            
        # Load FAISS index  
        if (load_dir / "faiss_index.bin").exists():  
            self.index = faiss.read_index(str(load_dir / "faiss_index.bin"))  
        else:  
            logger.warning("FAISS index file not found")  
            self.index = None  
            
        logger.info(f"Loaded KNN retriever from {directory} with {len(self.documents)} documents")  
    
    def print_stats(self) -> Dict[str, Any]:  
        """  
        Return statistics about the retriever.  
        
        Returns:  
            Dictionary of retriever statistics  
        """  
        return {  
            "name": self.name,  
            "type": self.__class__.__name__,  
            "document_count": len(self.documents),  
            "embedding_model": self.embedding_manager.embedding_model_name,  
            "index_type": self.index_type,  
            "score_threshold": self.score_threshold,  
            "dimension": self.dimension,  
            "faiss_is_trained": self.index and self.index.is_trained if self.index else False  
        }  







def run():
    print("=" * 60)
    print("KNN Retriever 测试")
    print("=" * 60)
    
    from langchain_core.documents import Document
    from config.settings import settings
    from providers.embedding_provider import EmbeddingProvider
    
    class MockEmbeddings:
        def __init__(self, dimension: int = 768):
            self.dimension = dimension
            
        def embed_query(self, text: str) -> list:
            import numpy as np
            np.random.seed(hash(text) % (2**32))
            return list(np.random.randn(self.dimension).astype(np.float64))
        
        def embed_documents(self, texts: list) -> list:
            return [self.embed_query(text) for text in texts]
    
    class MockEmbeddingManager:
        def __init__(self, dimension: int = 128):
            self.embedding_model_name = "mock_embeddings"
            self.dimension = dimension
            
        def embed_query(self, query: str) -> list:
            return self._get_vector(query)
            
        def embed_documents(self, documents: list) -> dict:
            result = {}
            for doc in documents:
                doc_id = doc.metadata.get("doc_id", str(len(result)))
                result[str(doc_id)] = self._get_vector(doc.page_content)
            return result
            
        def _get_vector(self, text: str) -> list:
            import numpy as np
            np.random.seed(abs(hash(text)) % (2**32))
            return list((np.random.randn(self.dimension) * 0.1).astype(np.float64))
    
    sample_documents = [
        Document(
            page_content="糖尿病是一种慢性代谢性疾病，其特征是血糖水平持续升高。",
            metadata={"doc_id": "1", "source": "medical_textbook", "category": "内分泌"}
        ),
        Document(
            page_content="高血压是指动脉血压持续升高，是心血管疾病的重要危险因素。",
            metadata={"doc_id": "2", "source": "medical_textbook", "category": "心血管"}
        ),
        Document(
            page_content="冠心病的全称是冠状动脉粥样硬化性心脏病。",
            metadata={"doc_id": "3", "source": "medical_guidelines", "category": "心血管"}
        ),
        Document(
            page_content="肺癌是全球发病率和死亡率最高的恶性肿瘤之一。",
            metadata={"doc_id": "4", "source": "oncology_research", "category": "肿瘤"}
        ),
        Document(
            page_content="2型糖尿病是最常见的糖尿病类型，与肥胖和生活方式密切相关。",
            metadata={"doc_id": "5", "source": "clinical_study", "category": "内分泌"}
        ),
        Document(
            page_content="脑卒中是急性脑血管疾病，是导致成年人残疾和死亡的主要原因。",
            metadata={"doc_id": "6", "source": "neuroscience", "category": "神经"}
        ),
        Document(
            page_content="肺炎是指肺实质的炎症，可由细菌或病毒引起。",
            metadata={"doc_id": "7", "source": "respiratory_medicine", "category": "呼吸"}
        ),
        Document(
            page_content="冠心病的治疗包括药物治疗、介入治疗和外科手术治疗。",
            metadata={"doc_id": "8", "source": "treatment_guidelines", "category": "心血管"}
        ),
    ]
    
    print(f"\n1. 初始化 Mock Embedding Manager...")
    mock_embedding_manager = MockEmbeddingManager(dimension=128)
    print(f"   嵌入维度: {mock_embedding_manager.dimension}")
    print(f"   模型名称: {mock_embedding_manager.embedding_model_name}")
    
    index_types = ["FLAT", "IVF", "HNSW"]
    
    for index_type in index_types:
        print(f"\n{'='*50}")
        print(f"测试索引类型: {index_type}")
        print(f"{'='*50}")
        
        print(f"\n2. 初始化 KNNRetriever ({index_type})...")
        from omegaconf import OmegaConf
        config = OmegaConf.create(
            name=f"test_{index_type.lower()}_retriever",
            score_threshold=0.1,
            index_type=index_type,
            n_list=4,
            m=16
        )
        retriever = KNNRetriever(config=config, embedding_manager=mock_embedding_manager)
        print(f"   检索器名称: {retriever.name}")
        print(f"   索引类型: {retriever.index_type}")
        print(f"   分数阈值: {retriever.score_threshold}")
        
        print(f"\n3. 添加文档...")
        retriever.add_documents(sample_documents)
        stats = retriever.print_stats()
        print(f"   文档数量: {stats['document_count']}")
        print(f"   向量维度: {stats['dimension']}")
        print(f"   索引已训练: {stats['faiss_is_trained']}")
        
        print(f"\n4. 测试搜索...")
        test_queries = [
            ("糖尿病", 2),
            ("冠心病", 2),
            ("心血管疾病", 2),
        ]
        
        for query, expected_min in test_queries:
            results = retriever.search(query, top_k=3)
            print(f"\n   查询: '{query}'")
            print(f"   结果数量: {len(results)}")
            for i, (doc, score) in enumerate(results[:2], 1):
                print(f"   [{i}] 分数: {score:.4f} | 类别: {doc.metadata.get('category', 'unknown')}")
                content_preview = doc.page_content[:40] + "..." if len(doc.page_content) > 40 else doc.page_content
                print(f"       内容: {content_preview}")
        
        print(f"\n5. 测试 HNSW 动态参数...")
        if index_type == "HNSW":
            results_fast = retriever.search("糖尿病治疗", top_k=3, ef_search=16)
            results_slow = retriever.search("糖尿病治疗", top_k=3, ef_search=200)
            print(f"   efSearch=16 结果数: {len(results_fast)}")
            print(f"   efSearch=200 结果数: {len(results_slow)}")
    
    print(f"\n{'='*50}")
    print("测试 FLAT 索引的删除功能")
    print(f"{'='*50}")
    
    from omegaconf import OmegaConf
    config = OmegaConf.create(name="delete_test", score_threshold=0.0, index_type="FLAT")
    retriever = KNNRetriever(config=config, embedding_manager=mock_embedding_manager)
    retriever.add_documents(sample_documents)
    
    print(f"\n6. 测试删除功能...")
    print(f"   删除前文档数量: {len(retriever.documents)}")
    retriever.delete_documents(["1", "2", "3"])
    print(f"   删除后文档数量: {len(retriever.documents)}")
    
    print(f"\n7. 验证删除后的搜索...")
    results = retriever.search("糖尿病", top_k=5)
    print(f"   查询'糖尿病'的结果数量: {len(results)}")
    
    print(f"\n8. 测试保存和加载...")
    save_dir = "/tmp/knn_test"
    retriever.save(save_dir)
    print(f"   已保存到: {save_dir}")
    
    from omegaconf import OmegaConf
    new_retriever = KNNRetriever(
        config=OmegaConf.create(name="loaded"),
        embedding_manager=mock_embedding_manager
    )
    new_retriever.load(save_dir)
    loaded_stats = new_retriever.print_stats()
    print(f"   加载后文档数量: {loaded_stats['document_count']}")
    print(f"   加载后维度: {loaded_stats['dimension']}")
    
    print(f"\n9. 测试过滤搜索...")
    filter_results = retriever.search_with_filter(
        query="心脏",
        filter_dict={"category": "心血管"},
        top_k=2
    )
    print(f"   查询'心脏' + 类别='心血管' 的结果数量: {len(filter_results)}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)







if __name__ == "__main__":
    run()