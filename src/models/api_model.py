"""  
API model implementation for Chinese-MedQA-Qwen2 project.  
This module provides wrappers for API-based model access, especially for Zhipuai API.  
"""  

import os  
import json  
import time  
from typing import Dict, List, Optional, Tuple, Union, Any  
import torch  
import numpy as np  
import requests  

from .base_model import BaseModel  


class ApiModel(BaseModel):  
    """  
    Base class for API-based models.  
    """  
    
    def __init__(  
        self,  
        api_key: str = None,  
        api_base: str = None,  
        api_type: str = "openai",  
        model_name: str = None,  
        device: str = "cpu", # Device parameter is kept for compatibility with BaseModel  
        **kwargs  
    ):  
        """  
        Initialize the API model.  
        
        Args:  
            api_key (str, optional): API key for the service  
            api_base (str, optional): Base URL for the API  
            api_type (str, optional): Type of API (openai, zhipuai, etc.)  
            model_name (str, optional): Name of the model to use  
            device (str, optional): Not used for API models, kept for compatibility  
            **kwargs: Additional model-specific parameters  
        """  
        super().__init__(model_path=model_name, device=device, **kwargs)  
        
        # Use environment variables if not provided  
        self.api_key = api_key or os.environ.get(f"{api_type.upper()}_API_KEY")  
        self.api_base = api_base or os.environ.get(f"{api_type.upper()}_API_BASE")  
        
        self.api_type = api_type  
        self.model_name = model_name  
        
        if not self.api_key:  
            raise ValueError(f"API key must be provided either as an argument or as {api_type.upper()}_API_KEY environment variable")  
    
    def load(self, **kwargs):  
        """  
        Load the API model. For API models, this is a no-op as there is no  
        local model to load, but we implement it to satisfy the BaseModel interface.  
        
        Returns:  
            Tuple: A tuple of (None, None) as there's no actual model or tokenizer  
        """  
        print(f"Using {self.api_type} API with model {self.model_name}")  
        # Return None for model and tokenizer as they don't exist locally  
        return None, None  
    
    def generate(self, prompt: str, **kwargs):  
        """  
        Generate text using the API.  
        
        Args:  
            prompt (str): The input prompt  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  
        raise NotImplementedError("Subclasses must implement this method")  
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs):  
        """  
        Get embeddings for the given texts using the API.  
        
        Args:  
            texts (Union[str, List[str]]): Text(s) to generate embeddings for  
            **kwargs: Additional parameters for embedding generation  
            
        Returns:  
            torch.Tensor: The generated embeddings  
        """  
        raise NotImplementedError("Subclasses must implement this method")  


class ZhipuAiModel(ApiModel):  
    """  
    Implementation for Zhipuai API using the official zhipuai Python library.  
    """  
    
    def __init__(  
        self,  
        api_key: str = None,  
        api_base: str = None,  # Not needed when using the SDK but kept for compatibility  
        model_name: str = "glm-4-flash",  
        **kwargs  
    ):  
        """  
        Initialize the Zhipuai model.  
        
        Args:  
            api_key (str, optional): Zhipuai API key  
            api_base (str, optional): Not used with the SDK but kept for compatibility  
            model_name (str, optional): Name of the model to use  
            **kwargs: Additional model-specific parameters  
        """  
        super().__init__(  
            api_key=api_key,  
            api_base=api_base,  
            api_type="zhipuai",  
            model_name=model_name,  
            **kwargs  
        )  
        
        try:  
            from zhipuai import ZhipuAI  
            self.client = ZhipuAI(api_key=self.api_key)  
            print("Successfully initialized zhipuai client")  
        except ImportError:  
            raise ImportError("zhipuai package is required. Install it with 'pip install zhipuai'")  
        
        # Maps for available models  
        self.available_llm_models = [  
            "glm-3-turbo",   
            "glm-4",   
            "glm-4-flash",
            "glm-4-turbo",
            "glm-4-vision",   
        ]  
        
        self.available_embedding_models = [  
            "embedding-2",   
            "embedding-3"
            "text_embedding"  
        ]  
        
        # Check if model is supported  
        if self.model_name not in self.available_llm_models and self.model_name not in self.available_embedding_models:  
            print(f"Warning: Model {self.model_name} not in known Zhipuai models. Using it anyway.")  
    
    def generate(  
        self,  
        prompt: str,  
        max_length: int = 2048,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        **kwargs  
    ) -> str:  
        """  
        Generate text using the Zhipuai API via the official SDK.  
        
        Args:  
            prompt (str): The input prompt  
            max_length (int): Maximum length of generated text  
            temperature (float): Sampling temperature  
            top_p (float): Nucleus sampling parameter  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  
        # Check if the prompt is already in messages format  
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):  
            messages = prompt  
        else:  
            # Format as a user message if it's just a string  
            messages = [{"role": "user", "content": prompt}]  
        
        # Prepare parameters for the API call  
        params = {  
            "model": self.model_name,  
            "messages": messages,  
            "max_tokens": max_length,  
            "temperature": temperature,  
            "top_p": top_p,  
        }  
        # Add any additional parameters  
        for key, value in kwargs.items():  
            if key not in params:  
                params[key] = value  
        
        # Make the API call using the zhipuai SDK  
        try:  
            response = self.client.chat.completions.create(**params)  
            
            # Extract the response text  
            if hasattr(response, 'choices') and len(response.choices) > 0:  
                return response.choices[0].message.content  
            elif isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:  
                return response['choices'][0]['message']['content']  
            else:  
                raise ValueError(f"Unexpected response format: {response}")  
                
        except Exception as e:  
            print(f"Zhipuai API request failed: {e}")  
            raise  
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:  
        """  
        Get embeddings for the given texts using the Zhipuai API via the official SDK.  
        
        Args:  
            texts (Union[str, List[str]]): Text(s) to generate embeddings for  
            **kwargs: Additional parameters for embedding generation  
            
        Returns:  
            torch.Tensor: The generated embeddings  
        """  
        # Ensure we're using an embedding model  
        embedding_model = kwargs.get("embedding_model", "embedding-2")  
        if embedding_model not in self.available_embedding_models:  
            print(f"Warning: Embedding model {embedding_model} not in known Zhipuai embedding models. Using it anyway.")  
        
        # Convert single text to list for consistent processing  
        if isinstance(texts, str):  
            texts = [texts]  
        
        all_embeddings = []  
        
        # Process each text  
        for text in texts:  
            # Prepare parameters for the API call  
            params = {  
                "model": embedding_model,  
                "input": text,  
            }  
            # Add any additional parameters except embedding_model  
            for key, value in kwargs.items():  
                if key != "embedding_model" and key not in params:  
                    params[key] = value  
            
            # Make the API call using the zhipuai SDK  
            try:  
                response = self.client.embeddings.create(**params)  
                
                # Extract embeddings  
                if hasattr(response, 'data') and len(response.data) > 0 and hasattr(response.data[0], 'embedding'):  
                    embedding = response.data[0].embedding  
                    all_embeddings.append(embedding)  
                elif isinstance(response, dict) and 'data' in response and len(response['data']) > 0 and 'embedding' in response['data'][0]:  
                    embedding = response['data'][0]['embedding']  
                    all_embeddings.append(embedding)  
                else:  
                    raise ValueError(f"Unexpected response format: {response}")  
                    
            except Exception as e:  
                print(f"Zhipuai API request failed: {e}")  
                raise  
            
            # Add a small delay to avoid rate limiting  
            time.sleep(0.1)  
        
        # Convert to torch tensor  
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)  
        return embeddings_tensor  
    
    def prepare_inputs_for_rag(self, query: str, context: List[str], **kwargs) -> Dict[str, Any]:  
        """  
        Prepare inputs for the RAG pipeline with Zhipuai API formatting.  
        
        Args:  
            query (str): The user query  
            context (List[str]): The retrieved context passages  
            **kwargs: Additional parameters  
            
        Returns:  
            Dict[str, Any]: The prepared inputs for the API  
        """  
        # Format context into a single string  
        context_str = "\n\n".join([f"[Document {i+1}]: {doc}" for i, doc in enumerate(context)])  
        
        # Create messages format  
        messages = [  
            {  
                "role": "system",   
                "content": "You are a helpful medical assistant that provides accurate information based on retrieved medical documents."  
            },  
            {  
                "role": "user",   
                "content": f"I need information about: {query}\n\nRelevant documents:\n{context_str}"  
            }  
        ]  
        
        return {  
            "messages": messages,  
            "query": query,  
            "context": context  
        }  


class OpenAiModel(ApiModel):  
    """  
    Implementation for OpenAI API.  
    """  
    
    def __init__(  
        self,  
        api_key: str = None,  
        api_base: str = "https://api.openai.com/v1",  
        model_name: str = "gpt-3.5-turbo",  
        **kwargs  
    ):  
        """  
        Initialize the OpenAI model.  
        
        Args:  
            api_key (str, optional): OpenAI API key  
            api_base (str, optional): Base URL for the OpenAI API  
            model_name (str, optional): Name of the model to use  
            **kwargs: Additional model-specific parameters  
        """  
        super().__init__(  
            api_key=api_key,  
            api_base=api_base,  
            api_type="openai",  
            model_name=model_name,  
            **kwargs  
        )  
        
        # Import here to avoid dependency issues if not using OpenAI  
        try:  
            import openai  
            self.client = openai.OpenAI(  
                api_key=self.api_key,  
                base_url=self.api_base  
            )  
        except ImportError:  
            print("Warning: openai package not installed. Using requests instead.")  
            self.client = None  
    
    def generate(  
        self,  
        prompt: str,  
        max_length: int = 2048,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        **kwargs  
    ) -> str:  
        """  
        Generate text using the OpenAI API.  
        
        Args:  
            prompt (str): The input prompt  
            max_length (int): Maximum length of generated text  
            temperature (float): Sampling temperature  
            top_p (float): Nucleus sampling parameter  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  
        # Check if the prompt is already in messages format  
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):  
            messages = prompt  
        else:  
            # Format as a user message if it's just a string  
            messages = [{"role": "user", "content": prompt}]  
        
        if self.client:  
            # Use the OpenAI client library if available  
            try:  
                response = self.client.chat.completions.create(  
                    model=self.model_name,  
                    messages=messages,  
                    max_tokens=max_length,  
                    temperature=temperature,  
                    top_p=top_p,  
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "model"]}  
                )  
                return response.choices[0].message.content  
            except Exception as e:  
                print(f"OpenAI API request failed: {e}")  
                raise  
        else:  
            # Fall back to using requests  
            headers = {  
                "Authorization": f"Bearer {self.api_key}",  
                "Content-Type": "application/json"  
            }  
            
            data = {  
                "model": self.model_name,  
                "messages": messages,  
                "max_tokens": max_length,  
                "temperature": temperature,  
                "top_p": top_p,  
                **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "model"]}  
            }  
            
            try:  
                response = requests.post(  
                    f"{self.api_base}/chat/completions",  
                    headers=headers,  
                    json=data,  
                    timeout=60  
                )  
                response.raise_for_status()  
                result = response.json()  
                
                if "choices" in result and len(result["choices"]) > 0:  
                    return result["choices"][0]["message"]["content"]  
                else:  
                    raise ValueError(f"Unexpected response format: {result}")  
                    
            except requests.exceptions.RequestException as e:  
                print(f"API request failed: {e}")  
                if hasattr(e, 'response') and e.response is not None:  
                    print(f"Response status: {e.response.status_code}")  
                    print(f"Response body: {e.response.text}")  
                raise  
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:  
        """  
        Get embeddings for the given texts using the OpenAI API.  
        
        Args:  
            texts (Union[str, List[str]]): Text(s) to generate embeddings for  
            **kwargs: Additional parameters for embedding generation  
            
        Returns:  
            torch.Tensor: The generated embeddings  
        """  
        # Ensure we're using an embedding model  
        embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")  
        
        # Convert single text to list for consistent processing  
        if isinstance(texts, str):  
            texts = [texts]  
        
        if self.client:  
            # Use the OpenAI client library if available  
            try:  
                response = self.client.embeddings.create(  
                    model=embedding_model,  
                    input=texts,  
                    **{k: v for k, v in kwargs.items() if k != "embedding_model"}  
                )  
                
                # Extract embeddings  
                embeddings = [item.embedding for item in response.data]  
                return torch.tensor(embeddings, dtype=torch.float32)  
                
            except Exception as e:  
                print(f"OpenAI API request failed: {e}")  
                raise  
        else:  
            # Fall back to using requests  
            headers = {  
                "Authorization": f"Bearer {self.api_key}",  
                "Content-Type": "application/json"  
            }  
            
            data = {  
                "model": embedding_model,  
                "input": texts,  
                **{k: v for k, v in kwargs.items() if k != "embedding_model"}  
            }  
            
            try:  
                response = requests.post(  
                    f"{self.api_base}/embeddings",  
                    headers=headers,  
                    json=data,  
                    timeout=60  
                )  
                response.raise_for_status()  
                result = response.json()  
                
                if "data" in result and len(result["data"]) > 0:  
                    embeddings = [item["embedding"] for item in result["data"]]  
                    return torch.tensor(embeddings, dtype=torch.float32)  
                else:  
                    raise ValueError(f"Unexpected response format: {result}")  
                    
            except requests.exceptions.RequestException as e:  
                print(f"API request failed: {e}")  
                if hasattr(e, 'response') and e.response is not None:  
                    print(f"Response status: {e.response.status_code}")  
                    print(f"Response body: {e.response.text}")  
                raise  
    
    def prepare_inputs_for_rag(self, query: str, context: List[str], **kwargs) -> Dict[str, Any]:  
        """  
        Prepare inputs for the RAG pipeline with OpenAI API formatting.  
        
        Args:  
            query (str): The user query  
            context (List[str]): The retrieved context passages  
            **kwargs: Additional parameters  
            
        Returns:  
            Dict[str, Any]: The prepared inputs for the API  
        """  
        # Format context into a single string  
        context_str = "\n\n".join([f"[Document {i+1}]: {doc}" for i, doc in enumerate(context)])  
        
        # Create messages format  
        messages = [  
            {  
                "role": "system",   
                "content": "You are a helpful medical assistant that provides accurate information based on retrieved medical documents."  
            },  
            {  
                "role": "user",   
                "content": f"I need information about: {query}\n\nRelevant documents:\n{context_str}"  
            }  
        ]  
        
        return {  
            "messages": messages,  
            "query": query,  
            "context": context  
        }  