# knowledge_base/retrieval/similarity_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os,sys
import numpy as np  
import logging  
import pickle  
from pathlib import Path  
import json  
import time  
from langchain_core.documents import Document  

from knowledge_base.retrieval.base_retriever import BaseRetriever  
from knowledge_base.embedding.embedding_manager import EmbeddingManager  
from config.settings import settings


class SimilarityRetriever(BaseRetriever):
    """
    Retriever that uses cosine similarity to find most similar documents.
    Implements a simple but effective vector similarity search.
    """

    def __init__(
        self,
        config=None,
        embedding_manager: EmbeddingManager=None,
    ):
        if config is None:
            config = settings.retriever
        if embedding_manager is None:
            from knowledge_base.embedding.embedding_manager import EmbeddingManager
            embedding_manager = EmbeddingManager()  
        """  
        Initialize the similarity retriever.  
        
        Args:  
            embedding_manager: Embedding manager for document and query embedding  
            name: Name of the retriever  
            score_threshold: Minimum similarity score threshold  
        """  
        super().__init__(config=config)  
        self.embedding_manager = embedding_manager  
        self.score_threshold = config.score_threshold  
        
        # Storage for documents and embeddings  
        self.documents: List[Document] = []  
        self.document_ids: List[str] = []  
        self.document_embeddings: List[np.ndarray] = []  
    
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        if not documents:  
            return  
            
        # Embed the documents  
        self.logger.info(f"Embedding {len(documents)} documents for similarity retriever")  
        start_time = time.time()  
        
        # Get document texts and IDs  
        doc_ids = []  
        for i, doc in enumerate(documents):  
            # Use document ID from metadata or generate one  
            doc_id = doc.metadata.get("doc_id", str(len(self.documents) + i))  
            doc_ids.append(str(doc_id))  
        
        # Embed the documents  
        try:  
            embeddings_dict = self.embedding_manager.embed_documents(documents)  
            
            # Add documents and embeddings to storage  
            for i, doc in enumerate(documents):  
                doc_id = doc_ids[i]  
                if doc_id in embeddings_dict:  
                    self.documents.append(doc)  
                    self.document_ids.append(doc_id)  
                    self.document_embeddings.append(np.array(embeddings_dict[doc_id]))  
                else:  
                    self.logger.warning(f"No embedding found for document {doc_id}")  
                    
            self.logger.info(f"Added {len(documents)} documents to similarity retriever in {time.time()-start_time:.2f}s")  
        except Exception as e:  
            self.logger.error(f"Error adding documents to similarity retriever: {str(e)}")  
            raise  
    
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
            self.document_embeddings.pop(idx)  
            
        self.logger.info(f"Deleted {len(indices_to_delete)} documents from similarity retriever")  
    
    def search(  
        self,   
        query: str,   
        top_k: int = 5,   
        score_threshold: Optional[float] = None  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search for documents similar to the query.  
        
        Args:  
            query: Query string  
            top_k: Number of top results to return  
            score_threshold: Minimum similarity score threshold (overrides instance threshold)  
            
        Returns:  
            List of (document, score) tuples  
        """  
        if not self.documents:  
            self.logger.warning("No documents in similarity retriever")  
            return []  
            
        # Use instance threshold if not specified  
        if score_threshold is None:  
            score_threshold = self.score_threshold  
            
        try:  
            # Embed the query  
            query_embedding = np.array(self.embedding_manager.embed_query(query))  
            
            # Calculate cosine similarity  
            doc_embeddings_matrix = np.vstack(self.document_embeddings)  
            
            # Normalize embeddings for cosine similarity  
            query_norm = np.linalg.norm(query_embedding)  
            doc_norms = np.linalg.norm(doc_embeddings_matrix, axis=1)  
            
            # Avoid division by zero  
            query_embedding = query_embedding / query_norm if query_norm > 0 else query_embedding  
            doc_embeddings_matrix = doc_embeddings_matrix / doc_norms[:, np.newaxis] if np.all(doc_norms > 0) else doc_embeddings_matrix  
            
            # Calculate similarities  
            similarities = np.dot(doc_embeddings_matrix, query_embedding)  
            
            # Get indices of top k results  
            top_indices = np.argsort(-similarities)[:min(top_k, len(similarities))]  
            
            # Build result list  
            results = []  
            for idx in top_indices:  
                score = similarities[idx]  
                if score >= score_threshold:  
                    results.append((self.documents[idx], float(score)))  
                    
            return results  
            
        except Exception as e:  
            self.logger.error(f"Error searching in similarity retriever: {str(e)}")  
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
            
        # Save document embeddings  
        with open(save_dir / "document_embeddings.pkl", "wb") as f:  
            pickle.dump(self.document_embeddings, f)  
            
        # Save config
        config = {  
            "name": self.config.name,  
            "score_threshold": self.score_threshold,  
            "embedding_model_name": self.embedding_manager.embedding_model_name,  
        }  
        with open(save_dir / "config.json", "w") as f:  
            json.dump(config, f)  
            
        self.logger.info(f"Saved similarity retriever to {directory}")  
    
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
            
        # Load document embeddings  
        with open(load_dir / "document_embeddings.pkl", "rb") as f:  
            self.document_embeddings = pickle.load(f)  
            
        # Load config  
        with open(load_dir / "config.json", "r") as f:  
            config = json.load(f)  
            self.name = config.get("name", self.name)  
            self.score_threshold = config.get("score_threshold", self.score_threshold)  
            
        self.logger.info(f"Loaded similarity retriever from {directory} with {len(self.documents)} documents")  
    
    def print_stats(self) -> Dict[str, Any]:  
        """  
        Return statistics about the retriever.  
        
        Returns:  
            Dictionary of retriever statistics  
        """  
        return {  
            "name": self.config.name,  
            "type": self.__class__.__name__,  
            "document_count": len(self.documents),  
            "embedding_model": self.embedding_manager.embedding_model_name,  
            "score_threshold": self.score_threshold  
        }  



def run():
    pass




if __name__ == "__main__":
    run()