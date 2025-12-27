# knowledge_base/embedding_manager.py  
from typing import List, Dict, Any, Optional, Union  
import os  
import numpy as np  
import logging  
from pathlib import Path  
import torch  
import json  
import time  
from langchain_core.documents import Document  
from langchain.embeddings.base import Embeddings  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings  
from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_community.embeddings import ModelScopeEmbeddings  
import pickle  

logger = logging.getLogger(__name__)  

class EmbeddingManager:  
    """  
    Manages text embeddings for the medical knowledge base.  
    Supports multiple embedding models and caching for efficiency.  
    """  
    
    def __init__(  
        self,  
        embedding_model_name: str = "m3e/m3e-base",  
        device: str = "cuda" if torch.cuda.is_available() else "cpu",  
        cache_dir: Optional[str] = None,  
        normalize_embeddings: bool = True,  
        use_cache: bool = True  
    ):  
        """  
        Initialize the embedding manager.  
        
        Args:  
            embedding_model_name: Name or path of the embedding model  
            device: Device to run the embedding model on  
            cache_dir: Directory to cache embeddings  
            normalize_embeddings: Whether to normalize embeddings  
            use_cache: Whether to use embedding caching  
        """  
        self.embedding_model_name = embedding_model_name  
        self.device = device  
        self.normalize_embeddings = normalize_embeddings  
        self.use_cache = use_cache  
        
        if cache_dir:  
            self.cache_dir = Path(cache_dir)  
            self.cache_dir.mkdir(parents=True, exist_ok=True)  
        else:  
            self.cache_dir = None  
            
        self.embedding_model = self._load_embedding_model()  
        self.embedding_cache = {}  
        
        # Load cache if it exists  
        if self.use_cache and self.cache_dir:  
            self._load_cache()  
    
    def _load_embedding_model(self) -> Embeddings:  
        """  
        Load the embedding model based on model name.  
        
        Returns:  
            Embeddings model instance  
        """  
        logger.info(f"Loading embedding model: {self.embedding_model_name}")  
        
        # Handle different model types  
        if "openai" in self.embedding_model_name.lower():  
            # Load OpenAI embeddings if API key is set  
            api_key = os.environ.get("OPENAI_API_KEY")  
            if not api_key:  
                raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI embeddings")  
            return OpenAIEmbeddings(model=self.embedding_model_name)  
            
        elif "bge" in self.embedding_model_name.lower():  
            # BGE models  
            return HuggingFaceBgeEmbeddings(  
                model_name=self.embedding_model_name,  
                model_kwargs={"device": self.device},  
                encode_kwargs={"normalize_embeddings": self.normalize_embeddings}  
            )  
            
        elif "modelscope" in self.embedding_model_name.lower():  
            # ModelScope models  
            return ModelScopeEmbeddings(  
                model_name=self.embedding_model_name  
            )  
            
        else:  
            # Default to HuggingFace models  
            return HuggingFaceEmbeddings(  
                model_name=self.embedding_model_name,  
                model_kwargs={"device": self.device},  
                encode_kwargs={"normalize_embeddings": self.normalize_embeddings}  
            )  
    
    def _get_cache_path(self) -> Path:  
        """  
        Get the path to the embedding cache file.  
        
        Returns:  
            Path to cache file  
        """  
        if not self.cache_dir:  
            raise ValueError("Cache directory not set")  
            
        model_name_safe = self.embedding_model_name.replace("/", "_")  
        return self.cache_dir / f"{model_name_safe}_embedding_cache.pkl"  
    
    def _load_cache(self) -> None:  
        """Load the embedding cache from disk if it exists."""  
        if not self.cache_dir:  
            return  
            
        cache_path = self._get_cache_path()  
        if cache_path.exists():  
            try:  
                with open(cache_path, "rb") as f:  
                    self.embedding_cache = pickle.load(f)  
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings from {cache_path}")  
            except Exception as e:  
                logger.warning(f"Failed to load embedding cache: {str(e)}")  
                self.embedding_cache = {}  
    
    def _save_cache(self) -> None:  
        """Save the embedding cache to disk."""  
        if not self.use_cache or not self.cache_dir:  
            return  
            
        cache_path = self._get_cache_path()  
        try:  
            with open(cache_path, "wb") as f:  
                pickle.dump(self.embedding_cache, f)  
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache at {cache_path}")  
        except Exception as e:  
            logger.warning(f"Failed to save embedding cache: {str(e)}")  
    
    def _get_cache_key(self, text: str) -> str:  
        """  
        Generate a cache key for a text.  
        
        Args:  
            text: Text to generate cache key for  
            
        Returns:  
            Cache key string  
        """  
        # Use a hash of the text as the cache key  
        return str(hash(text))  
    
    def embed_query(self, query: str) -> List[float]:  
        """  
        Embed a query string.  
        
        Args:  
            query: Query string to embed  
            
        Returns:  
            Embedding vector as a list of floats  
        """  
        if not query.strip():  
            raise ValueError("Query cannot be empty")  
            
        cache_key = self._get_cache_key(query)  
        
        # Check cache first  
        if self.use_cache and cache_key in self.embedding_cache:  
            return self.embedding_cache[cache_key]  
            
        # Get embedding from model  
        try:  
            embedding = self.embedding_model.embed_query(query)  
            
            # Cache the result  
            if self.use_cache:  
                self.embedding_cache[cache_key] = embedding  
                
            return embedding  
        except Exception as e:  
            logger.error(f"Error embedding query: {str(e)}")  
            raise  
    
    def embed_documents(self, documents: List[Document]) -> Dict[str, List[float]]:  
        """  
        Embed a list of documents.  
        
        Args:  
            documents: List of documents to embed  
            
        Returns:  
            Dictionary mapping document IDs to embeddings  
        """  
        if not documents:  
            return {}  
            
        # Texts to embed, filtering out empty documents  
        valid_docs = [(i, doc) for i, doc in enumerate(documents) if doc.page_content.strip()]  
        
        if not valid_docs:  
            logger.warning("No valid documents to embed")  
            return {}  
            
        indices, docs_to_embed = zip(*valid_docs)  
        texts = [doc.page_content for doc in docs_to_embed]  
        
        # Check cache for each document  
        embeddings = []  
        texts_to_embed = []  
        uncached_indices = []  
        
        if self.use_cache:  
            for i, text in enumerate(texts):  
                cache_key = self._get_cache_key(text)  
                if cache_key in self.embedding_cache:  
                    embeddings.append((indices[i], self.embedding_cache[cache_key]))  
                else:  
                    texts_to_embed.append(text)  
                    uncached_indices.append(indices[i])  
        else:  
            texts_to_embed = texts  
            uncached_indices = indices  
        
        # Embed uncached texts  
        if texts_to_embed:  
            try:  
                start_time = time.time()  
                uncached_embeddings = self.embedding_model.embed_documents(texts_to_embed)  
                logger.info(f"Embedded {len(texts_to_embed)} documents in {time.time()-start_time:.2f}s")  
                
                # Cache the results  
                if self.use_cache:  
                    for i, text, embedding in zip(uncached_indices, texts_to_embed, uncached_embeddings):  
                        cache_key = self._get_cache_key(text)  
                        self.embedding_cache[cache_key] = embedding  
                        embeddings.append((i, embedding))  
                else:  
                    embeddings.extend(zip(uncached_indices, uncached_embeddings))  
                    
            except Exception as e:  
                logger.error(f"Error embedding documents: {str(e)}")  
                raise  
        
        # Sort embeddings by original indices  
        embeddings.sort(key=lambda x: x[0])  
        
        # Create result dictionary with document ids as keys  
        result = {}  
        for i, (_, embedding) in enumerate(embeddings):  
            doc_id = documents[indices[i]].metadata.get("doc_id", str(i))  
            result[str(doc_id)] = embedding  
            
        # Save cache after embedding  
        if self.use_cache:  
            self._save_cache()  
            
        return result  
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:  
        """  
        Embed a list of texts in batches.  
        
        Args:  
            texts: List of texts to embed  
            batch_size: Batch size for embedding  
            
        Returns:  
            List of embedding vectors  
        """  
        if not texts:  
            return []  
            
        result = []  
        
        for i in range(0, len(texts), batch_size):  
            batch = texts[i:i+batch_size]  
            try:  
                # Check cache first  
                batch_embeddings = []  
                texts_to_embed = []  
                uncached_indices = []  
                
                if self.use_cache:  
                    for j, text in enumerate(batch):  
                        cache_key = self._get_cache_key(text)  
                        if cache_key in self.embedding_cache:  
                            batch_embeddings.append((j, self.embedding_cache[cache_key]))  
                        else:  
                            texts_to_embed.append(text)  
                            uncached_indices.append(j)  
                else:  
                    texts_to_embed = batch  
                    uncached_indices = list(range(len(batch)))  
                
                # Embed uncached texts  
                if texts_to_embed:  
                    uncached_embeddings = self.embedding_model.embed_documents(texts_to_embed)  
                    
                    # Cache the results  
                    if self.use_cache:  
                        for j, text, embedding in zip(uncached_indices, texts_to_embed, uncached_embeddings):  
                            cache_key = self._get_cache_key(text)  
                            self.embedding_cache[cache_key] = embedding  
                            batch_embeddings.append((j, embedding))  
                    else:  
                        batch_embeddings.extend(zip(uncached_indices, uncached_embeddings))  
                
                # Sort embeddings by original indices  
                batch_embeddings.sort(key=lambda x: x[0])  
                result.extend([emb for _, emb in batch_embeddings])  
                
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")  
                
            except Exception as e:  
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")  
                raise  
                
        return result  