# knowledge_base/retrieval/bm25_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os,sys
import logging  
import pickle  
from pathlib import Path  
import json  
import time  
import numpy as np  
from langchain_core.documents import Document  
import jieba  
from rank_bm25 import BM25Okapi  

sys.path.append(str(Path(__file__).parent.parent.parent))

from knowledge_base.retrieval.base_retriever import BaseRetriever  
from config.retriever_config import BM25RetrieverConfig

logger = logging.getLogger(__name__)  

class BM25Retriever(BaseRetriever):  
    """  
    Retriever that uses BM25 algorithm to find relevant documents.  
    BM25 is a bag-of-words retrieval function that ranks documents based on   
    the query terms appearing in each document.  
    """  
    
    def __init__(  
        self,  
        name: str = "bm25_retriever",  
        score_threshold: float = 0.1,  
        use_jieba: bool = True,  
        tokenizer: Optional[callable] = None  
    ):  
        """  
        Initialize the BM25 retriever.  
        
        Args:  
            name: Name of the retriever  
            score_threshold: Minimum BM25 score threshold  
            use_jieba: Whether to use jieba for Chinese tokenization  
            tokenizer: Custom tokenizer function  
        """  
        super().__init__(name=name)  
        self.score_threshold = score_threshold  
        self.use_jieba = use_jieba  
        self.tokenizer = tokenizer  
        
        # Storage for documents and BM25 model  
        self.documents: List[Document] = []  
        self.document_ids: List[str] = []  
        self.bm25_model = None  
        self.tokenized_corpus = []  
    
    def _tokenize(self, text: str) -> List[str]:  
        """  
        Tokenize text using appropriate tokenizer.  
        
        Args:  
            text: Text to tokenize  
            
        Returns:  
            List of tokens  
        """  
        if self.tokenizer:  
            return self.tokenizer(text)  
            
        if self.use_jieba:  
            # For Chinese text  
            return [token for token in jieba.cut(text) if token.strip()]  
        else:  
            # Simple whitespace tokenization for non-Chinese text  
            return [token for token in text.lower().split() if token.strip()]  
    
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        if not documents:  
            return  
            
        start_time = time.time()  
        logger.info(f"Adding {len(documents)} documents to BM25 retriever")  
        
        # Add documents to storage  
        for i, doc in enumerate(documents):  
            doc_id = doc.metadata.get("doc_id", str(len(self.documents) + i))  
            self.documents.append(doc)  
            self.document_ids.append(str(doc_id))  
            
        # Tokenize all documents  
        corpus = [doc.page_content for doc in self.documents]  
        self.tokenized_corpus = [self._tokenize(text) for text in corpus]  
        
        # Create BM25 model  
        self.bm25_model = BM25Okapi(self.tokenized_corpus)  
        
        logger.info(f"Added {len(documents)} documents to BM25 retriever in {time.time()-start_time:.2f}s")  
    
    def delete_documents(self, document_ids: List[str]) -> None:  
        """  
        Delete documents from the retriever's index.  
        
        Args:  
            document_ids: List of document IDs to delete  
        """  
        if not document_ids or not self.documents:  
            return  
            
        # Find indices of documents to delete  
        indices_to_delete = []  
        for i, doc_id in enumerate(self.document_ids):  
            if doc_id in document_ids:  
                indices_to_delete.append(i)  
                
        # Delete in reverse order to avoid index shifting  
        for idx in sorted(indices_to_delete, reverse=True):  
            self.documents.pop(idx)  
            self.document_ids.pop(idx)  
            
        # Recreate BM25 model with remaining documents  
        if self.documents:  
            corpus = [doc.page_content for doc in self.documents]  
            self.tokenized_corpus = [self._tokenize(text) for text in corpus]  
            self.bm25_model = BM25Okapi(self.tokenized_corpus)  
        else:  
            self.tokenized_corpus = []  
            self.bm25_model = None  
            
        logger.info(f"Deleted {len(indices_to_delete)} documents from BM25 retriever")  
    
    def search(  
        self,   
        query: str,   
        top_k: int = 5,   
        score_threshold: Optional[float] = None  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search for documents similar to the query using BM25.  
        
        Args:  
            query: Query string  
            top_k: Number of top results to return  
            score_threshold: Minimum BM25 score threshold (overrides instance threshold)  
            
        Returns:  
            List of (document, score) tuples  
        """  
        if not self.documents or not self.bm25_model:  
            logger.warning("No documents in BM25 retriever")  
            return []  
            
        # Use instance threshold if not specified  
        if score_threshold is None:  
            score_threshold = self.score_threshold  
            
        try:  
            # Tokenize query  
            tokenized_query = self._tokenize(query)  
            
            # Get BM25 scores  
            scores = self.bm25_model.get_scores(tokenized_query)  
            
            # Get indices of top k results  
            top_indices = np.argsort(-scores)[:min(top_k, len(scores))]  
            
            # Build result list  
            results = []  
            for idx in top_indices:  
                score = scores[idx]  
                if score >= score_threshold:  
                    results.append((self.documents[idx], float(score)))  
                    
            return results  
            
        except Exception as e:  
            logger.error(f"Error searching in BM25 retriever: {str(e)}")  
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
            
        # Save tokenized corpus  
        with open(save_dir / "tokenized_corpus.pkl", "wb") as f:  
            pickle.dump(self.tokenized_corpus, f)  
            
        # Save BM25 model  
        with open(save_dir / "bm25_model.pkl", "wb") as f:  
            pickle.dump(self.bm25_model, f)  
            
        # Save config  
        config = {  
            "name": self.name,  
            "score_threshold": self.score_threshold,  
            "use_jieba": self.use_jieba  
        }  
        with open(save_dir / "config.json", "w") as f:  
            json.dump(config, f)  
            
        logger.info(f"Saved BM25 retriever to {directory}")  
    
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
            
        # Load tokenized corpus  
        with open(load_dir / "tokenized_corpus.pkl", "rb") as f:  
            self.tokenized_corpus = pickle.load(f)  
            
        # Load BM25 model  
        with open(load_dir / "bm25_model.pkl", "rb") as f:  
            self.bm25_model = pickle.load(f)  
            
        # Load config  
        with open(load_dir / "config.json", "r") as f:  
            config = json.load(f)  
            self.name = config.get("name", self.name)  
            self.score_threshold = config.get("score_threshold", self.score_threshold)  
            self.use_jieba = config.get("use_jieba", self.use_jieba)  
            
        logger.info(f"Loaded BM25 retriever from {directory} with {len(self.documents)} documents")  
    
    def print_stats(self) -> Dict[str, Any]:  
        """  
        Return statistics about the retriever.  
        
        Returns:  
            Dictionary of retriever statistics  
        """  
        vocab_size = len(self.bm25_model.idf) if self.bm25_model else 0  
        
        return {  
            "name": self.name,  
            "type": self.__class__.__name__,  
            "document_count": len(self.documents),  
            "vocabulary_size": vocab_size,  
            "score_threshold": self.score_threshold,  
            "tokenizer": "jieba" if self.use_jieba else "whitespace"  
        }  





def run():
    print("=" * 60)
    print("BM25 Retriever 测试")
    print("=" * 60)
    
    from langchain_core.documents import Document
    
    sample_documents = [
        Document(
            page_content="糖尿病是一种慢性代谢性疾病，其特征是血糖水平持续升高。主要类型包括1型糖尿病和2型糖尿病。",
            metadata={"doc_id": "1", "source": "medical_textbook", "category": "内分泌"}
        ),
        Document(
            page_content="高血压是指动脉血压持续升高，收缩压≥140mmHg或舒张压≥90mmHg。是心血管疾病的重要危险因素。",
            metadata={"doc_id": "2", "source": "medical_textbook", "category": "心血管"}
        ),
        Document(
            page_content="冠心病的全称是冠状动脉粥样硬化性心脏病，是由于冠状动脉发生粥样硬化导致心肌缺血缺氧而引起的心脏病。",
            metadata={"doc_id": "3", "source": "medical_guidelines", "category": "心血管"}
        ),
        Document(
            page_content="肺癌是全球发病率和死亡率最高的恶性肿瘤之一。主要危险因素包括吸烟、空气污染和职业性暴露。",
            metadata={"doc_id": "4", "source": "oncology_research", "category": "肿瘤"}
        ),
        Document(
            page_content="2型糖尿病是最常见的糖尿病类型，通常与胰岛素抵抗和胰岛β细胞功能缺陷有关，与肥胖和生活方式密切相关。",
            metadata={"doc_id": "5", "source": "clinical_study", "category": "内分泌"}
        ),
        Document(
            page_content="脑卒中是急性脑血管疾病，包括缺血性脑卒中和出血性脑卒中，是导致成年人残疾和死亡的主要原因之一。",
            metadata={"doc_id": "6", "source": "neuroscience", "category": "神经"}
        ),
        Document(
            page_content="肺炎是指肺实质的炎症，可由细菌、病毒、真菌等病原体引起。常见症状包括发热、咳嗽和呼吸困难。",
            metadata={"doc_id": "7", "source": "respiratory_medicine", "category": "呼吸"}
        ),
        Document(
            page_content="冠心病的治疗包括药物治疗、介入治疗和外科手术治疗。常用药物包括阿司匹林、他汀类和β受体阻滞剂。",
            metadata={"doc_id": "8", "source": "treatment_guidelines", "category": "心血管"}
        ),
    ]
    
    print(f"\n1. 初始化 BM25 检索器...")
    retriever = BM25Retriever(
        name="medical_bm25_retriever",
        score_threshold=0.1,
        use_jieba=True
    )
    print(f"   检索器名称: {retriever.name}")
    print(f"   分数阈值: {retriever.score_threshold}")
    print(f"   使用分词器: {'jieba' if retriever.use_jieba else 'whitespace'}")
    
    print(f"\n2. 添加文档到索引...")
    retriever.add_documents(sample_documents)
    stats = retriever.print_stats()
    print(f"   文档数量: {stats['document_count']}")
    print(f"   词汇表大小: {stats['vocabulary_size']}")
    
    print(f"\n3. 测试搜索功能...")
    test_queries = [
        "糖尿病的治疗方法",
        "冠心病的症状",
        "肺癌的危险因素",
        "高血压的定义",
    ]
    
    for query in test_queries:
        print(f"\n   查询: '{query}'")
        results = retriever.search(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"   [{i}] 分数: {score:.4f} | 来源: {doc.metadata.get('source', 'unknown')} | 类别: {doc.metadata.get('category', 'unknown')}")
            content_preview = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            print(f"       内容: {content_preview}")
    
    print(f"\n4. 测试分词功能...")
    test_text = "糖尿病和高血压是常见的慢性疾病"
    tokens = retriever._tokenize(test_text)
    print(f"   原文: {test_text}")
    print(f"   分词结果: {tokens}")
    
    print(f"\n5. 测试删除功能...")
    print(f"   删除前文档数量: {len(retriever.documents)}")
    retriever.delete_documents(["1", "2"])
    print(f"   删除后文档数量: {len(retriever.documents)}")
    
    print(f"\n6. 验证删除后的搜索结果...")
    results = retriever.search("糖尿病", top_k=3)
    print(f"   查询'糖尿病'的结果数量: {len(results)}")
    
    print(f"\n7. 测试保存和加载功能...")
    save_dir = "/tmp/bm25_test"
    retriever.save(save_dir)
    print(f"   已保存到: {save_dir}")
    
    new_retriever = BM25Retriever(name="loaded_retriever")
    new_retriever.load(save_dir)
    loaded_stats = new_retriever.print_stats()
    print(f"   加载后文档数量: {loaded_stats['document_count']}")
    
    print(f"\n8. 测试过滤搜索...")
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