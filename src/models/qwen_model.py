"""  
Qwen2 model implementation for Chinese-MedQA-Qwen2 project.  
"""  

import os  
import torch  
from typing import Dict, List, Optional, Tuple, Union, Any  

from transformers import (  
    AutoTokenizer,   
    AutoModelForCausalLM,   
    BitsAndBytesConfig,  
    PreTrainedModel,   
    PreTrainedTokenizer  
)  

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseGenerativeModel  
from config.local_model_config import LocalModelConfig


# from models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import (
    Qwen3ForCausalLM
)

class Qwen2Model(BaseGenerativeModel):  
    """  
    Qwen2 model implementation for the Chinese-MedQA-Qwen2 project.  
    
    This class implements the BaseModel interface for Qwen2 models, supporting  
    different model sizes and configurations.  
    """  
    
    def __init__(  
        self,  
        model_path: str = "Qwen/Qwen2-7B",  
        device: str = None,  
        load_in_8bit: bool = False,  
        load_in_4bit: bool = False,  
        use_flash_attention: bool = False,  
        max_context_length: int = 8192,  
        trust_remote_code: bool = True,  
        **kwargs  
    ):  
        """  
        Initialize the Qwen2 model.  
        
        Args:  
            model_path (str): Path or identifier of the Qwen2 model  
            device (str, optional): Device to run the model on  
            load_in_8bit (bool): Whether to quantize the model to 8 bits  
            load_in_4bit (bool): Whether to quantize the model to 4 bits  
            use_flash_attention (bool): Whether to use flash attention for faster inference  
            max_context_length (int): Maximum context length for the model  
            trust_remote_code (bool): Whether to trust remote code from model repository  
            **kwargs: Additional model-specific parameters  
        """  
        super().__init__(model_path=model_path, device=device, **kwargs)  
        
        self.load_in_8bit = load_in_8bit  
        self.load_in_4bit = load_in_4bit  
        self.use_flash_attention = use_flash_attention  
        self.max_context_length = max_context_length  
        self.trust_remote_code = trust_remote_code  
        
        # Load model and tokenizer if model_path is provided  
        if self.model_path:  
            self.model, self.tokenizer = self.load()  
    
    def load(self, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        print(f"Loading Qwen2 model from {self.model_path}")
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            **kwargs
        )
        
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device_map = "auto" if self.device != "cpu" else None
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
            **kwargs
        )
        
        if self.device == "cpu":
            model = model.to(self.device)
        
        print(f"Model loaded successfully")
        return model, tokenizer  
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation")
        
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template") and isinstance(prompt, list)
        
        if use_chat_template and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages = prompt
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, device=self.device)
            }
        else:
            if isinstance(prompt, list):
                prompt = "\n".join(prompt)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"],
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        if use_chat_template:
            response = self._extract_assistant_response(outputs[0])
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if isinstance(prompt, str) and response.startswith(prompt):
                response = response[len(prompt):].strip()
        
        return response
    
    def _extract_assistant_response(self, output_ids: torch.Tensor) -> str:
        """Extract assistant response from model output using tokenizer."""
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
                if response.startswith(":"):
                    response = response[1:].strip()
        
        return response  
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:  
        """  
        Get embeddings for the given texts using the Qwen2 model.  
        Note: For text embeddings, typically a dedicated embedding model should be used.  
        This method provides a simple implementation using the model's hidden states.  
        
        Args:  
            texts (Union[str, List[str]]): Text(s) to generate embeddings for  
            **kwargs: Additional parameters for embedding generation  
            
        Returns:  
            torch.Tensor: The generated embeddings  
        """  
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model and tokenizer must be loaded before getting embeddings")  
        
        # Convert single text to list for consistent processing  
        if isinstance(texts, str):  
            texts = [texts]  
        
        embeddings = []  
        
        for text in texts:  
            # Tokenize the text  
            inputs = self.tokenizer(  
                text,  
                return_tensors="pt",  
                padding=True,  
                truncation=True,  
                max_length=self.max_context_length  
            ).to(self.device)  
            
            # Get the model's last hidden states  
            with torch.no_grad():  
                outputs = self.model(  
                    input_ids=inputs.input_ids,  
                    attention_mask=inputs.attention_mask,  
                    output_hidden_states=True,  
                    **kwargs  
                )  
            
            # Use the last hidden state of the last token as the embedding  
            last_hidden_state = outputs.hidden_states[-1]  
            # Get the embedding of the last token for each sequence  
            seq_embed = last_hidden_state[:, -1, :]  
            embeddings.append(seq_embed)  
        
        # Stack all embeddings  
        return torch.cat(embeddings, dim=0)  # shape = (n*bsz, hidden_size)
    
    def prepare_inputs_for_rag(self, query: str, context: List[str], **kwargs) -> Dict[str, Any]:  
        """  
        Prepare inputs for the RAG pipeline with Qwen2 formatting.  
        
        Args:  
            query (str): The user query  
            context (List[str]): The retrieved context passages  
            **kwargs: Additional parameters  
            
        Returns:  
            Dict[str, Any]: The prepared inputs for the model  
        """  
        # Format context into a single string  
        context_str = "\n\n".join([f"[Document {i+1}]: {doc}" for i, doc in enumerate(context)])  
        
        # Create chat-like format  
        if hasattr(self.tokenizer, "apply_chat_template"):  
            messages = [  
                {"role": "system", "content": "You are a helpful medical assistant that provides accurate information based on retrieved medical documents."},  
                {"role": "user", "content": f"I need information about: {query}\n\nRelevant documents:\n{context_str}"}  
            ]  
            
            return {  
                "messages": messages,  
                "query": query,  
                "context": context  
            }  
        else:  
            # For non-chat models, construct a prompt manually  
            prompt = f"""  
            Please answer the following medical question based on the provided reference documents.  
            
            Question: {query}  
            
            Reference documents:  
            {context_str}  
            
            Answer:  
            """  
            
            return {  
                "prompt": prompt,  
                "query": query,  
                "context": context  
            }  


class Qwen2ForMedicalQA(Qwen2Model):
    MEDICAL_SAFETY_WARNING = (
        "⚠️ 重要医疗免责声明\n"
        "本系统提供的医疗信息仅供参考，不能替代专业医疗建议、诊断或治疗。\n"
        "请务必咨询合格的医疗专业人员以获得准确的医疗建议。\n"
        "如果您遇到紧急医疗情况，请立即拨打当地急救电话或前往最近的医疗机构。"
    )
    
    SAFETY_KEYWORDS = [
        "紧急", "急救", "急诊", "拨打", "医生", "医院", "专业医疗",
        "emergency", "urgent", "call doctor", "medical professional"
    ]
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-7B-Instruct",
        device: str = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        max_context_length: int = 8192,
        trust_remote_code: bool = True,
        require_safety_confirmation: bool = True,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            max_context_length=max_context_length,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        self.require_safety_confirmation = require_safety_confirmation
    
    def generate_medical_answer(
        self,
        query: str,
        context: List[str],
        max_length: int = 1024,
        temperature: float = 0.5,
        include_safety_warning: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation")
        
        safety_flag = self._check_medical_safety(query)
        
        inputs = self.prepare_inputs_for_rag(query, context)
        
        if "messages" in inputs:
            messages = inputs["messages"]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
        else:
            prompt = inputs["prompt"]
        
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            **kwargs
        )
        
        result = {
            "query": query,
            "response": response,
            "safety_flag": safety_flag,
            "context_sources": len(context) if context else 0
        }
        
        if include_safety_warning:
            result["safety_warning"] = self.MEDICAL_SAFETY_WARNING
        
        return result
    
    def _check_medical_safety(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        is_urgent = any(keyword in query for keyword in self.SAFETY_KEYWORDS)
        
        return {
            "is_potentially_urgent": is_urgent,
            "requires_professional_review": is_urgent,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_medical_response(self, response: str, query: str, **kwargs) -> Dict[str, Any]:
        validation_prompt = f"""作为医疗信息质量评估助手，请评估以下回答的质量。

问题：{query}

回答：{response}

请从以下维度评估回答质量（只需简单评价，无需详细分析）：
1. 是否有帮助性
2. 是否包含不当医疗建议
3. 是否建议咨询专业人士

评估结果："""
        
        evaluation = self.generate(
            validation_prompt,
            max_new_tokens=150,
            temperature=0.3
        )
        
        concerns = []
        response_lower = response.lower()
        
        urgent_keywords = ["紧急", "立即", "马上", "急诊", "emergency", "immediately"]
        if any(kw in response_lower for kw in urgent_keywords):
            concerns.append("Contains urgent medical keywords - verify context")
        
        disclaimer_keywords = ["咨询医生", "专业医疗", "请就医", "consult", "professional"]
        if not any(kw in response_lower for kw in disclaimer_keywords):
            concerns.append("Missing professional consultation recommendation")
        
        return {
            "response": response,
            "evaluation": evaluation,
            "concerns": concerns,
            "is_safe": len(concerns) == 0,
            "safety_warning": self.MEDICAL_SAFETY_WARNING if concerns else None
        }  
        
        





class ModelLoadingError(Exception):
    pass

class GenerationError(Exception):
    pass

class InputValidationError(Exception):
    pass


def validate_medical_query(query: str, max_length: int = 1000) -> bool:
    if not query or not query.strip():
        raise InputValidationError("查询内容不能为空")
    
    if len(query) > max_length:
        raise InputValidationError(f"查询内容过长，最大支持{max_length}个字符")
    
    return True


def run():
    try:
        local_model_config = LocalModelConfig(
            model_path=f"{str(Path(__file__).parent.parent.parent)}/../models/Qwen3-0.6B",
            device="cpu"
        )
        
        print("正在初始化模型...")
        model = Qwen2Model(
            model_path=local_model_config.model_path, 
            device=local_model_config.device,
            load_in_4bit=True,
        )
        print("模型初始化完成")
        
        test_prompt = "今天天气很不错，你觉得呢？"
        print(f"测试提示词: {test_prompt}")
        
        result = model.generate(test_prompt)
        
        print("\n生成结果:")
        print(result)
        
        return result
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    run()
    
   