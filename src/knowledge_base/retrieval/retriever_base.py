# knowledge_base/retrieval/retriever_base.py  
from abc import ABC, abstractmethod  
from typing import List, Dict, Any, Optional, Tuple  
from langchain.schema import Document  

class BaseRetriever(ABC):  
    """  
    Base class for all retrievers in the medical knowledge base.  
    All retrieval methods should inherit from this class.  
    """  
    
    def __init__(self, name: str):  
        """  
        Initialize the base retriever.  
        
        Args:  
            name: Name of the retriever  
        """  
        self.name = name  
    
    @abstractmethod  
    def add_documents(self, documents: List[Document]) -> None:  
        """  
        Add documents to the retriever's index.  
        
        Args:  
            documents: List of documents to add  
        """  
        pass  
    
    @abstractmethod  
    def delete_documents(self, document_ids: List[str]) -> None:  
        """  
        Delete documents from the retriever's index.  
        
        Args:  
            document_ids: List of document IDs to delete  
        """  
        pass  
    
    @abstractmethod  
    def search(  
        self,   
        query: str,   
        top_k: int = 5,   
        **kwargs  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search for documents similar to the query.  
        
        Args:  
            query: Query string  
            top_k: Number of top results to return  
            **kwargs: Additional search parameters  
            
        Returns:  
            List of (document, score) tuples  
        """  
        pass  
    
    @abstractmethod  
    def save(self, directory: str) -> None:  
        """  
        Save the retriever to a directory.  
        
        Args:  
            directory: Directory to save to  
        """  
        pass  
    
    @abstractmethod  
    def load(self, directory: str) -> None:  
        """  
        Load the retriever from a directory.  
        
        Args:  
            directory: Directory to load from  
        """  
        pass  
    
    def search_with_filter(  
        self,   
        query: str,   
        filter_dict: Dict[str, Any],   
        top_k: int = 5,   
        **kwargs  
    ) -> List[Tuple[Document, float]]:  
        """  
        Search with filtering by metadata fields.  
        
        Args:  
            query: Query string  
            filter_dict: Dictionary of metadata fields to filter by  
            top_k: Number of top results to return  
            **kwargs: Additional search parameters  
            
        Returns:  
            List of (document, score) tuples  
        """  
        # Default implementation: search then filter  
        results = self.search(query, top_k=top_k * 2, **kwargs)  # Get more results to allow for filtering  
        
        filtered_results = []  
        for doc, score in results:  
            matches_all = True  
            for key, value in filter_dict.items():  
                if key not in doc.metadata or doc.metadata[key] != value:  
                    matches_all = False  
                    break    # 如果一个文档片段没有包含所有的过滤条件，那就把他丢掉
                    
            if matches_all:  
                filtered_results.append((doc, score))  
                
            if len(filtered_results) >= top_k:  
                break  
                
        return filtered_results  
    
    def print_stats(self) -> Dict[str, Any]:  
        """  
        Return statistics about the retriever.  
        
        Returns:  
            Dictionary of retriever statistics  
        """  
        # Default implementation, should be overridden by subclasses  
        return {"name": self.name, "type": self.__class__.__name__}  