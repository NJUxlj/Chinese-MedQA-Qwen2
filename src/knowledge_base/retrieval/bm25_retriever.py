# knowledge_base/retrieval/bm25_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os  
import logging  
import pickle  
from pathlib import Path  
import json  
import time  
import numpy as np  
from langchain.schema import Document  
import jieba  
from rank_bm25 import BM25Okapi  

from knowledge_base.retrieval.retriever_base import BaseRetriever  

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