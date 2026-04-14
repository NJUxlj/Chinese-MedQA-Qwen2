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
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod

from models.base_model import BaseGenerativeModel  
from config.llm_config import LLMConfig
from zhipuai import ZhipuAI  
import openai  

from utils.logger import setup_logger
from retrying import retry



class ApiNameClassMapping:
    pass



class ApiModel(ABC):  
    """  
    Base class for API-based models.  
    """  
    
    def __init__(  
        self,  
        llm_config: LLMConfig,  
        **kwargs  
    ):  
        """  
        Initialize the API model.  
        
        Args:  
            llm_config (LLMConfig): The configuration for the LLM model.  
            **kwargs: Additional model-specific parameters  
        """ 
        
        # Use environment variables if not provided  
        self.llm_config = llm_config
        self.logger = setup_logger(self.__class__.__name__)
        
        if not self.llm_config.api_key:  
            raise ValueError(f"API key must be provided either as an argument or as {self.llm_config.model_provider().upper()}_API_KEY environment variable")  
    
    
    def generate(self, prompt: str, additional_args={}, messages = None,**kwargs)->str:  
        """  
        Generate text using the API.  
        
        Args:  
            prompt (str): The input prompt  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  

        args = {
            "stream": self.llm_config.stream,
            "temperature": self.llm_config.temperature,
            "top_p": self.llm_config.top_p,
            "max_tokens": self.llm_config.max_tokens,
        }

        if self.llm_config.model_name != 'gpt-5':
            args["top_k"] = self.llm_config.top_k

        args.update(additional_args)
        
        # 为gpt-5模型处理参数名称差异和限制
        if self.llm_config.model_name == 'gpt-5':
            # 将max_tokens转换为max_completion_tokens
            if 'max_tokens' in args:
                args['max_completion_tokens'] = args.pop('max_tokens')
            
            # gpt-5模型只支持默认的temperature值(1)，移除自定义temperature
            if 'temperature' in args and args['temperature'] != 1:
                args.pop('temperature')
            
            # gpt-5模型可能不支持top_p参数，移除它
            if 'top_p' in args:
                args.pop('top_p')
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.llm_config.api_key
            }

        else:

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.api_key}"
            }



        if messages is None:
            messages = [{
                'role': 'user',
                'content': prompt
            }]
        else:
            messages = messages
        
        payload = {
            "model": self.llm_config.model_name,
            "messages": messages,
            **args,
        }

        extra_body = additional_args.get('extra_body') if isinstance(additional_args, dict) else None
        if extra_body and isinstance(extra_body, dict) and any(k in self.llm_config.model_name for k in ['qwen', 'Qwen']):
            payload['extra_body'] = extra_body

        try:
            try:
                total_chars = sum(len(m.get('content', '')) for m in messages)
            except Exception:
                total_chars = -1

            # 添加调试信息
            self.logger.info(f"Sending LLM API request to : {self.llm_config.base_url}")
            self.logger.info(f"API Model Name: {self.llm_config.model_name}")
            self.logger.info(f"Total messages characters: chars={total_chars}")
            self.logger.info("Waiting API server response ...")
            
            request_timeout = min(self.llm_config.timeout, 200)
            response = requests.post(self.llm_config.base_url, headers=headers, json=payload, timeout=request_timeout)

            self.logger.info("Request is sent successfully...")

            self.logger.info(f"response code: {response.status_code}\n\n")
            
            if response.status_code != 200:
                self.logger.error(f"Response content: {response.text}\n\n")
                response.raise_for_status()

            response_data: dict = response.json()

            if 'error' in response_data:
                raise ValueError(f"API Error: {response_data}")
            

            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError(f"API Error: Empty choices: {response_data}")

            choice = response_data["choices"][0]


            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]

            if "delta" in choice and "content" in choice["delta"]:
                return choice["delta"]["content"]

            raise ValueError(f"No content in response: {response_data}")


        except requests.exceptions.Timeout:
            raise ValueError(f"API calling timeout ({request_timeout}秒): {self.llm_config.model_name}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API calling request error: {e}")
        except Exception as e:
            raise ValueError(f"API calling exception: {e}")
    
    @retry(wait_fixed=3000, stop_max_attempt_number=2)
    def retry_generate(self, prompt: str, additional_args={}, messages = None,**kwargs):
        return self.generate(prompt, additional_args, messages, **kwargs)
    
 






class ZhipuApiModel(ApiModel):  
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


class OpenAIApiModel(BaseGenerativeModel):  
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
                base_url=self.base_url  
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
                    f"{self.base_url}/chat/completions",  
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
                    f"{self.base_url}/embeddings",  
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








class QwenApiModel(BaseGenerativeModel):  
    """  
    Implementation for OpenAI API.  
    """  
    
    def __init__(  
        self,  
        api_key: str = None,  
        base_url: str = "https://aliyun.com/v1",  
        model_name: str = "qwen3",  
        **kwargs  
    ):  
        """  
        Initialize the OpenAI model.  
        
        Args:  
            api_key (str, optional): OpenAI API key  
            base_url (str, optional): Base URL for the OpenAI API  
            model_name (str, optional): Name of the model to use  
            **kwargs: Additional model-specific parameters  
        """  
        super().__init__(  
            api_key=api_key,  
            base_url=base_url,  
            api_type="qwen",  
            model_name=model_name,  
            **kwargs  
        )  
        
        # Import here to avoid dependency issues if not using OpenAI  
        try:  
            self.client = openai.OpenAI(  
                api_key=self.api_key,  
                base_url=self.base_url  
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
                    f"{self.base_url}/chat/completions",  
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
                    f"{self.base_url}/embeddings",  
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