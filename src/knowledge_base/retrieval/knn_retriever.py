# knowledge_base/retrieval/knn_retriever.py  
from typing import List, Dict, Any, Optional, Tuple  
import os  
import numpy as np  
import logging  
import pickle  
from pathlib import Path  
import json  
import time  
import faiss  
from langchain.schema import Document  

from knowledge_base.retrieval.retriever_base import BaseRetriever  
from knowledge_base.embedding_manager import EmbeddingManager  

logger = logging.getLogger(__name__)  

class KNNRetriever(BaseRetriever):  
    """  
    Retriever that uses KNN with FAISS for efficient similarity search.  
    FAISS provides fast and scalable similarity search.  
    """  
    
    def __init__(  
        self,  
        embedding_manager: EmbeddingManager,  
        name: str = "knn_retriever",  
        score_threshold: float = 0.5,  
        index_type: str = "Flat",  # "Flat", "IVF", "HNSW"  
        n_list: int = 100,         # For IVF index  
        m: int = 16                # For HNSW index  
    ):  
        """  
        Initialize the KNN retriever.  
        
        Args:  
            embedding_manager: Embedding manager for document and query embedding  
            name: Name of the retriever  
            score_threshold: Minimum similarity score threshold  
            index_type: FAISS index type  
            n_list: Number of IVF clusters (for IVF index)  
            m: Number of connections per element (for HNSW index)  
        """  
        super().__init__(name=name)  
        self.embedding_manager = embedding_manager  
        self.score_threshold = score_threshold  
        self.index_type = index_type  
        self.n_list = n_list  
        self.m = m  
        
        # Storage for documents  
        self.documents: List[Document] = []  
        self.document_ids: List[str] = []  
        
        # FAISS index  
        self.index = None  
        self.dimension = None  
    
    def _create_index(self, dimension: int) -> faiss.Index:  
        """  
        Create a FAISS index with the specified parameters.  
        
        Args:  
            dimension: Dimensionality of the vectors  
            
        Returns:  
            FAISS index  
        """  
        if self.index_type == "Flat":  
            # Simple but exact index  
            return faiss.IndexFlatIP(dimension)  
            
        elif self.index_type == "IVF":  
            # More efficient but approximate index  
            quantizer = faiss.IndexFlatIP(dimension)  
            return faiss.IndexIVFFlat(quantizer, dimension, self.n_list, faiss.METRIC_INNER_PRODUCT)  
            
        elif self.index_type == "HNSW":  
            # Fast and memory-efficient index  
            index = faiss.IndexHNSWFlat(dimension, self.m, faiss.METRIC_INNER_PRODUCT)  
            index.hnsw.efConstruction = 200  
            index.hnsw.efSearch = 128  
            return index  
            
        else:  
            raise ValueError(f"Unknown index type: {self.index_type}")  
    
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        if not documents:  
            return  
            
        # Embed the documents  
        logger.info(f"Embedding {len(documents)} documents for KNN retriever")  
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
            
            # Create vectors array for FAISS  
            vectors = []  
            valid_docs = []  
            valid_ids = []  
            
            for i, doc in enumerate(documents):  
                doc_id = doc_ids[i]  
                if doc_id in embeddings_dict:  
                    vector = np.array(embeddings_dict[doc_id]).astype('float32')  
                    
                    # Normalize vector for inner product  
                    faiss.normalize_L2(vector.reshape(1, -1))  
                    
                    vectors.append(vector)  
                    valid_docs.append(doc)  
                    valid_ids.append(doc_id)  
                    
            if not vectors:  
                logger.warning("No valid embeddings to add to KNN retriever")  
                return  
                
            # Convert to numpy array  
            vectors_array = np.vstack(vectors)  
            
            # Initialize index if needed  
            if self.index is None:  
                self.dimension = vectors_array.shape[1]  
                self.index = self._create_index(self.dimension)  
                
                # Train if needed  
                if self.index_type == "IVF":  
                    logger.info("Training IVF index")  
                    self.index.train(vectors_array)  
            
            # Add vectors to index  
            self.index.add(vectors_array)  
            
            # Add documents to storage  
            self.documents.extend(valid_docs)  
            self.document_ids.extend(valid_ids)  
            
            logger.info(f"Added {len(valid_docs)} documents to KNN retriever in {time.time()-start_time:.2f}s")  
            
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
        if not self.documents or self.index is None:  
            logger.warning("No documents in KNN retriever or index not initialized")  
            return []  
            
        # Use instance threshold if not specified  
        if score_threshold is None:  
            score_threshold = self.score_threshold  
            
        try:  
            # Embed the query  
            query_embedding = np.array(self.embedding_manager.embed_query(query)).astype('float32')  
            
            # Normalize query embedding for inner product  
            faiss.normalize_L2(query_embedding.reshape(1, -1))  
            
            # Search in FAISS index  
            scores, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k, len(self.documents)))  
            
            # Build result list  
            results = []  
            for i, idx in enumerate(indices[0]):  
                if idx != -1:  # -1 means no result found  
                    score = float(scores[0][i])  
                    if score >= score_threshold:  
                        results.append((self.documents[idx], score))  
                    
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
            "name": self.name,  
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