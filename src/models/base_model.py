"""  
Base model class for Chinese-MedQA-Qwen2 project.  
This module provides the abstract base class for all models.  
"""  

import os  
from abc import ABC, abstractmethod  
from typing import Dict, List, Optional, Tuple, Union, Any  

import torch  
from transformers import PreTrainedModel, PreTrainedTokenizer  


class BaseModel(ABC):  
    """  
    Abstract base class for all models in the Chinese-MedQA-Qwen2 project.  
    
    This class defines the common interface that all model implementations  
    must adhere to, ensuring consistency across different model types.  
    """  
    
    def __init__(self, model_path: str = None, device: str = None, **kwargs):  
        """  
        Initialize the base model.  
        
        Args:  
            model_path (str, optional): Path to the model weights or model identifier  
            device (str, optional): Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.)  
            **kwargs: Additional model-specific parameters  
        """  
        self.model_path = model_path  
        
        # Determine device if not specified  
        if device is None:  
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        else:  
            self.device = device  
            
        # Placeholders for model and tokenizer  
        self.model = None  
        self.tokenizer = None  
        self.model_config = None  
    
    @abstractmethod  
    def load(self, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:  
        """  
        Load the model and tokenizer.  
        
        Args:  
            **kwargs: Additional parameters for loading  
            
        Returns:  
            Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer  
        """  
        pass  
    
    @abstractmethod  
    def generate(self,   
                 prompt: str,   
                 max_length: int = 512,   
                 temperature: float = 0.7,   
                 top_p: float = 0.9,   
                 top_k: int = 50,  
                 **kwargs) -> str:  
        """  
        Generate text based on the given prompt.  
        
        Args:  
            prompt (str): The input prompt for text generation  
            max_length (int, optional): Maximum length of generated text  
            temperature (float, optional): Sampling temperature  
            top_p (float, optional): Nucleus sampling parameter  
            top_k (int, optional): Top-k sampling parameter  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  
        pass  
    
    @abstractmethod  
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:  
        """  
        Get embeddings for the given texts.  
        
        Args:  
            texts (Union[str, List[str]]): Text(s) to generate embeddings for  
            **kwargs: Additional parameters for embedding generation  
            
        Returns:  
            torch.Tensor: The generated embeddings  
        """  
        pass  
    
    def prepare_inputs_for_rag(self, query: str, context: List[str], **kwargs) -> Dict[str, Any]:  
        """  
        Prepare inputs for the RAG pipeline.  
        
        Args:  
            query (str): The user query  
            context (List[str]): The retrieved context passages  
            **kwargs: Additional parameters  
            
        Returns:  
            Dict[str, Any]: The prepared inputs for the model  
        """  
        # Default implementation to be overridden by subclasses if needed  
        return {"query": query, "context": context}  
    
    def save(self, output_dir: str) -> None:  
        """  
        Save the model and tokenizer.  
        
        Args:  
            output_dir (str): Directory to save the model to  
            
        Returns:  
            None  
        """  
        if self.model is not None and self.tokenizer is not None:  
            os.makedirs(output_dir, exist_ok=True)  
            self.model.save_pretrained(output_dir)  
            self.tokenizer.save_pretrained(output_dir)  
            print(f"Model and tokenizer saved to {output_dir}")  
        else:  
            raise ValueError("Model and tokenizer must be loaded before saving")  
    
    def __repr__(self) -> str:  
        """String representation of the model"""  
        return f"{self.__class__.__name__}(model_path={self.model_path}, device={self.device})"  