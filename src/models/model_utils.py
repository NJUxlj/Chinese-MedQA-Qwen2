"""  
Utility functions for model handling in Chinese-MedQA-Qwen2 project.  
"""  

import os  
import json  
import torch  
from typing import Dict, List, Optional, Tuple, Union, Any  
import numpy as np  
from pathlib import Path  


def get_available_devices() -> Dict[str, Any]:  
    """  
    Get information about available devices for model execution.  
    
    Returns:  
        Dict[str, Any]: Information about available devices  
    """  
    devices = {  
        "cpu": {  
            "available": True,  
            "memory": "N/A"  
        }  
    }  
    
    # Check for CUDA (NVIDIA GPUs)  
    if torch.cuda.is_available():  
        devices["cuda"] = {  
            "available": True,  
            "count": torch.cuda.device_count(),  
            "devices": []  
        }  
        
        for i in range(torch.cuda.device_count()):  
            devices["cuda"]["devices"].append({  
                "index": i,  
                "name": torch.cuda.get_device_name(i),  
                "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB",  
                "memory_available": f"{torch.cuda.mem_get_info(i)[0] / (1024**3):.2f} GB"  
            })  
    
    # Check for MPS (Apple Silicon)  
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  
        devices["mps"] = {  
            "available": True,  
            "device": "Apple Silicon"  
        }  
    
    return devices  


def select_best_device() -> str:  
    """  
    Select the best available device for model execution.  
    
    Returns:  
        str: Name of the best available device  
    """  
    if torch.cuda.is_available():  
        return "cuda"  
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  
        return "mps"  
    else:  
        return "cpu"  


def calculate_model_size_requirements(  
    model_name: str,  
    precision: str = "fp16",  
    with_safety_margin: bool = True  
) -> Dict[str, float]:  
    """  
    Calculate the memory requirements for a model.  
    
    Args:  
        model_name (str): Name of the model  
        precision (str): Precision mode ('fp16', 'fp32', 'int8', 'int4')  
        with_safety_margin (bool): Whether to add a safety margin  
        
    Returns:  
        Dict[str, float]: Memory requirements in different formats  
    """  
    # Extract model size from name  
    model_size_map = {  
        "Qwen/Qwen2-7B": 7,  
        "Qwen/Qwen2-7B-Instruct": 7,  
        "Qwen/Qwen2-0.5B": 0.5,  
        "Qwen/Qwen2-1.5B": 1.5,  
        "Qwen/Qwen2-1.5B-Instruct": 1.5,  
        "Qwen/Qwen2-72B": 72,  
        "Qwen/Qwen2-72B-Instruct": 72,  
        
        "Qwen/Qwen2.5-7B": 7,
        "Qwen/Qwen2.5-7B-Instruct": 7,
        "Qwen/Qwen2.5-0.5B": 0.5,
        "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
        "Qwen/Qwen2.5-1.5B": 1.5,
        "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
        "Qwen/Qwen2.5-72B": 72,
        "Qwen/Qwen2.5-72B-Instruct": 72,
    }  
    
    # Default to 7B if unknown  
    model_size = None  
    for key, size in model_size_map.items():  
        if key in model_name:  
            model_size = size  
            break  
    
    if model_size is None:  
        # Try to extract from name (e.g., 7B, 13B)  
        import re  
        match = re.search(r'(\d+)B', model_name)  
        if match:  
            model_size = float(match.group(1))  
        else:  
            # Default size  
            model_size = 7.0  
    
    # Base memory requirements (fp32)  
    params_size_gb = model_size * 4 / 1024  
    
    # Adjust for precision  
    if precision == "fp16" or precision == "bf16":  
        params_size_gb /= 2  
    elif precision == "int8":  
        params_size_gb /= 4  
    elif precision == "int4":  
        params_size_gb /= 8  
    
    # Calculate KV cache, activations, and gradients (if training)  
    # Simple heuristic: KV cache + activations ~ 1.5x model size  
    additional_memory = params_size_gb * 1.5  
    
    # Total memory requirements  
    total_memory = params_size_gb + additional_memory  
    
    # Add safety margin (50%)  
    if with_safety_margin:  
        total_memory *= 1.5  
    
    return {  
        "model_params_gb": params_size_gb,  
        "total_memory_gb": total_memory,  
        "formatted_total": f"{total_memory:.2f} GB",  
        "model_size_billions": model_size  
    }  


def save_model_config(  
    model_config: Dict[str, Any],  
    output_dir: str,  
    filename: str = "model_config.json"  
) -> str:  
    """  
    Save model configuration to a file.  
    
    Args:  
        model_config (Dict[str, Any]): Model configuration  
        output_dir (str): Directory to save the configuration to  
        filename (str): Filename for the configuration  
        
    Returns:  
        str: Path to the saved configuration  
    """  
    os.makedirs(output_dir, exist_ok=True)  
    
    output_path = os.path.join(output_dir, filename)  
    with open(output_path, 'w', encoding='utf-8') as f:  
        json.dump(model_config, f, indent=2, ensure_ascii=False)  
    
    return output_path  


def load_model_config(config_path: str) -> Dict[str, Any]:  
    """  
    Load model configuration from a file.  
    
    Args:  
        config_path (str): Path to the configuration file  
        
    Returns:  
        Dict[str, Any]: Model configuration  
    """  
    with open(config_path, 'r', encoding='utf-8') as f:  
        return json.load(f)  


def get_model_class(model_type: str, **kwargs):  
    """  
    Get the appropriate model class based on the model type.  
    
    Args:  
        model_type (str): Type of model to load  
        **kwargs: Additional parameters  
        
    Returns:  
        class: The model class to use  
    """  
    from .base_model import BaseModel  
    from .qwen_model import Qwen2Model, Qwen2ForMedicalQA  
    from .api_model import ZhipuAiModel, OpenAiModel  
    
    model_classes = {  
        "qwen2": Qwen2Model,  
        "qwen2-medical": Qwen2ForMedicalQA,  
        "zhipuai": ZhipuAiModel,  
        "openai": OpenAiModel,  
        # Add more model types as needed  
    }  
    
    if model_type not in model_classes:  
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")  
    
    return model_classes[model_type]  


def model_factory(config: Dict[str, Any]):  
    """  
    Factory function to create model instances based on configuration.  
    
    Args:  
        config (Dict[str, Any]): Model configuration  
        
    Returns:  
        BaseModel: An instance of the appropriate model  
    """  
    model_type = config.pop("model_type", "qwen2")  
    model_class = get_model_class(model_type)  
    
    return model_class(**config)  


def ensure_model_compatibility(model_type: str, task: str) -> bool:  
    """  
    Check if a model is compatible with a task.  
    
    Args:  
        model_type (str): Type of model  
        task (str): Task to use the model for  
        
    Returns:  
        bool: Whether the model is compatible with the task  
    """  
    # Define task compatibility for different model types  
    compatibility = {  
        "qwen2": ["generation", "chat", "medical_qa", "embedding"],  
        "qwen2-medical": ["medical_qa", "chat", "generation"],  
        "zhipuai": ["chat", "generation", "medical_qa", "embedding"],  
        "openai": ["chat", "generation", "medical_qa", "embedding"],  
    }  
    
    if model_type not in compatibility:  
        return False  
    
    return task in compatibility[model_type]  


class ModelRegistry:  
    """  
    Registry for managing and accessing models.  
    """  
    
    def __init__(self, models_dir: str = None):  
        """  
        Initialize the model registry.  
        
        Args:  
            models_dir (str, optional): Directory for model configurations  
        """  
        self.models = {}  
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "configs")  
        os.makedirs(self.models_dir, exist_ok=True)  
        self._load_registered_models()  
    
    def _load_registered_models(self):  
        """Load registered models from the models directory."""  
        for config_file in Path(self.models_dir).glob("*.json"):  
            try:  
                with open(config_file, 'r', encoding='utf-8') as f:  
                    config = json.load(f)  
                model_id = config.get("model_id", config_file.stem)  
                self.models[model_id] = config  
            except Exception as e:  
                print(f"Error loading model config from {config_file}: {e}")  
    
    def register_model(self, model_config: Dict[str, Any]) -> str:  
        """  
        Register a model with the registry.  
        
        Args:  
            model_config (Dict[str, Any]): Model configuration  
            
        Returns:  
            str: ID of the registered model  
        """  
        model_id = model_config.get("model_id", f"model_{len(self.models)}")  
        model_config["model_id"] = model_id  
        self.models[model_id] = model_config  
        
        # Save to file  
        config_path = os.path.join(self.models_dir, f"{model_id}.json")  
        with open(config_path, 'w', encoding='utf-8') as f:  
            json.dump(model_config, f, indent=2, ensure_ascii=False)  
        
        return model_id  
    
    def get_model(self, model_id: str, **kwargs):  
        """  
        Get a model instance by ID.  
        
        Args:  
            model_id (str): ID of the model to get  
            **kwargs: Additional parameters to override the configuration  
            
        Returns:  
            BaseModel: An instance of the model  
        """  
        if model_id not in self.models:  
            raise ValueError(f"Model {model_id} not found in registry")  
        
        config = self.models[model_id].copy()  
        config.update(kwargs)  
        
        return model_factory(config)  
    
    def list_models(self) -> List[Dict[str, Any]]:  
        """  
        List all registered models.  
        
        Returns:  
            List[Dict[str, Any]]: List of model configurations  
        """  
        return list(self.models.values())  
    
    def unregister_model(self, model_id: str) -> bool:  
        """  
        Unregister a model from the registry.  
        
        Args:  
            model_id (str): ID of the model to unregister  
            
        Returns:  
            bool: Whether the operation was successful  
        """  
        if model_id not in self.models:  
            return False  
        
        del self.models[model_id]  
        
        # Remove from file  
        config_path = os.path.join(self.models_dir, f"{model_id}.json")  
        if os.path.exists(config_path):  
            os.remove(config_path)  
        
        return True  


def create_model_from_config(config_path: str):  
    """  
    Create a model from a configuration file.  
    
    Args:  
        config_path (str): Path to the configuration file  
        
    Returns:  
        BaseModel: An instance of the model  
    """  
    config = load_model_config(config_path)  
    return model_factory(config)  