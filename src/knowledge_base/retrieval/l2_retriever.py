# knowledge_base/retrieval/l2_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os,sys  
import numpy as np  
import logging  
import pickle  
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent.parent))
import json  
import time  
from langchain_core.documents import Document  

from knowledge_base.retrieval.base_retriever import BaseRetriever  
from knowledge_base.embedding.embedding_manager import EmbeddingManager  
from utils.logger import setup_logger
from config.retriever_config import L2RetrieverConfig

logger = logging.getLogger(__name__)  

class L2Retriever(BaseRetriever):  
    """  
    Retriever that uses L2 Euclidean distance to find most similar documents.  
    Lower distances indicate higher similarity.  
    """  
    
    def __init__(  
        self,  
        config: L2RetrieverConfig,
        embedding_manager: EmbeddingManager,  
    ):  
        """  
        Initialize the L2 retriever.  
        
        Args:  
            embedding_manager: Embedding manager for document and query embedding  
            name: Name of the retriever  
            distance_threshold: Maximum L2 distance threshold (lower is better)  
        """  
        super().__init__(config=config)  
        self.embedding_manager = embedding_manager  
        self.distance_threshold = config.distance_threshold  
        
        # Storage for documents and embeddings  
        self.documents: List[Document] = []  
        self.document_ids: List[str] = []  
        self.document_embeddings: List[np.ndarray] = [] 

        self.logger = setup_logger() 
    
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        if not documents:  
            return  
            
        # Embed the documents  
        logger.info(f"Embedding {len(documents)} documents for L2 retriever")  
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
                    logger.warning(f"No embedding found for document {doc_id}")  
                    
            logger.info(f"Added {len(documents)} documents to L2 retriever in {time.time()-start_time:.2f}s")  
        except Exception as e:  
            logger.error(f"Error adding documents to L2 retriever: {str(e)}")  
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
            
        logger.info(f"Deleted {len(indices_to_delete)} documents from L2 retriever")  
    
    def search(  
        self,   
        query: str,   
        top_k: int = 5,   
        distance_threshold: Optional[float] = None  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search for documents similar to the query.  
        
        Args:  
            query: Query string  
            top_k: Number of top results to return  
            distance_threshold: Maximum L2 distance threshold (overrides instance threshold)  
            
        Returns:  
            List of (document, score) tuples  
        """  
        if not self.documents:  
            logger.warning("No documents in L2 retriever")  
            return []  
            
        # Use instance threshold if not specified  
        if distance_threshold is None:  
            distance_threshold = self.distance_threshold  
            
        try:  
            # Embed the query  
            query_embedding = np.array(self.embedding_manager.embed_query(query))  
            
            # Calculate L2 distances  
            distances = []  
            for emb in self.document_embeddings:  
                distance = np.linalg.norm(query_embedding - emb)  
                distances.append(distance)  
            
            # Get indices of top k results (smallest distances)  
            top_indices = np.argsort(distances)[:min(top_k, len(distances))]  
            
            # Build result list  
            results = []  
            for idx in top_indices:  
                distance = distances[idx]  
                if distance <= distance_threshold:  
                    # Convert distance to similarity score (1 - normalized distance)  
                    # This makes higher scores better, similar to other retrievers  
                    max_distance = np.sqrt(2)  # Max L2 distance for unit vectors  
                    similarity = 1 - (distance / max_distance)  
                    results.append((self.documents[idx], float(similarity)))  
                    
            return results  
            
        except Exception as e:  
            logger.error(f"Error searching in L2 retriever: {str(e)}")  
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
            "name": self.name,  
            "distance_threshold": self.distance_threshold,  
            "embedding_model_name": self.embedding_manager.embedding_model_name,  
        }  
        with open(save_dir / "config.json", "w") as f:  
            json.dump(config, f)  
            
        logger.info(f"Saved L2 retriever to {directory}")  
    
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
            self.distance_threshold = config.get("distance_threshold", self.distance_threshold)  
            
        logger.info(f"Loaded L2 retriever from {directory} with {len(self.documents)} documents")  
    
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
            "distance_threshold": self.distance_threshold  
        }  






def run():
    pass





if __name__ == '__main__':
    run()



