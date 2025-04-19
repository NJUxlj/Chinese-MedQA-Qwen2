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

from .base_model import BaseModel  


from .qwen2.modeling_qwen2 import Qwen2ForCausalLM

class Qwen2Model(BaseModel):  
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
        use_flash_attention: bool = True,  
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
        """  
        Load the Qwen2 model and tokenizer.  
        
        Args:  
            **kwargs: Additional parameters for loading  
            
        Returns:  
            Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer  
        """  
        print(f"Loading Qwen2 model from {self.model_path}")  
        
        # Set up quantization configuration  
        quantization_config = None  
        if self.load_in_4bit:  
            # 大幅减少模型内存占用（约减少 75%）
            quantization_config = BitsAndBytesConfig(  
                load_in_4bit=True,  
                bnb_4bit_compute_dtype=torch.bfloat16,  
                bnb_4bit_use_double_quant=True,    # 启用双重量化，即对量化参数本身再进行一次量化
                bnb_4bit_quant_type="nf4"    # 使用 NF4 (Normal Float 4) 量化类型，这是一种专门为神经网络权重设计的 4-bit 量化格式
            )  
        elif self.load_in_8bit:  
            quantization_config = BitsAndBytesConfig(  
                load_in_8bit=True  
            )  
        
        # Load tokenizer  
        tokenizer = AutoTokenizer.from_pretrained(  
            self.model_path,  
            trust_remote_code=self.trust_remote_code,  
            **kwargs  
        )  
        
        # Ensure the tokenizer has padding token  
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        # Load model  
        model = AutoModelForCausalLM.from_pretrained(  
            self.model_path,  
            device_map=self.device if self.device != "cpu" else None,  
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,  
            quantization_config=quantization_config,  
            trust_remote_code=self.trust_remote_code,  
            attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",  
            **kwargs  
        )  
        
        # Move model to device if CPU  
        if self.device == "cpu":  
            model = model.to(self.device)  
        
        print(f"Model loaded on {model.device}")  
        return model, tokenizer  
    
    def generate(  
        self,  
        prompt: str,  
        max_length: int = 512,
        max_new_tokens: int = 200,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        top_k: int = 50,  
        num_return_sequences: int = 1,  
        do_sample: bool = True,  
        **kwargs  
    ) -> str:  
        """  
        Generate text based on the given prompt using the Qwen2 model.  
        
        Args:  
            prompt (str): The input prompt for text generation  
            max_length (int): Maximum length of generated text  
            temperature (float): Sampling temperature  
            top_p (float): Nucleus sampling parameter  
            top_k (int): Top-k sampling parameter  
            num_return_sequences (int): Number of sequences to return  
            do_sample (bool): Whether to use sampling  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated text  
        """  
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model and tokenizer must be loaded before generation")  
        
        # Add system message if supported by the model  
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template")  
        
        if use_chat_template:  
            messages = [  
                {"role": "system", "content": "You are a helpful medical assistant that provides accurate information based on your knowledge."},  
                {"role": "user", "content": prompt}  
            ]  
            input_ids = self.tokenizer.apply_chat_template(  
                messages,   
                return_tensors="pt",  
                add_generation_prompt=True  
            ).to(self.device) 
            
            # 确保inputs是字典格式
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)
            } 
        else:  
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  
        
        # Generate output  
        with torch.no_grad():  
            outputs = self.model.generate(  
                inputs["input_ids"],  
                max_length=max_length,
                max_new_tokens = max_new_tokens,  
                temperature=temperature,  
                top_p=top_p,  
                top_k=top_k,  
                num_return_sequences=num_return_sequences,   # 控制模型返回的候选序列数量。当值大于1时，模型会返回多个可能的输出序列（beam search）。
                do_sample=do_sample,  
                pad_token_id=self.tokenizer.pad_token_id,  
                attention_mask=inputs["attention_mask"] if use_chat_template else inputs.attention_mask,  
                **kwargs  
            )  
            
            '''
            当 do_sample=True 时:
                计算每个token的概率分布
                根据temperature调整分布平滑度
                使用top_k/top_p过滤低概率token
                从剩余token中随机采样
            当 do_sample=False 时：
                直接选择概率最高的token
                相当于temperature=0的确定式生成
            '''
        
        # Decode the output  
        if use_chat_template:  
            # Get only the assistant's response  
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            # Extract the assistant's message  
            assistant_response = decoded.split("assistant")[-1].strip()  
            if assistant_response.startswith(":"):  
                assistant_response = assistant_response[1:].strip()  
            return assistant_response  
        else:  
            # For non-chat models, just decode the output  
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            # Try to find and remove the prompt from the output  
            if decoded.startswith(prompt):  
                return decoded[len(prompt):].strip()  
            return decoded  
    
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
    """  
    Specialized Qwen2 model for medical question answering.  
    Extends the base Qwen2Model with medical-specific functionality.  
    """  
    
    def __init__(  
        self,  
        model_path: str = "Qwen/Qwen2-7B-Instruct",  
        device: str = None,  
        load_in_8bit: bool = False,  
        load_in_4bit: bool = False,  
        use_flash_attention: bool = True,  
        max_context_length: int = 8192,  
        trust_remote_code: bool = True,  
        **kwargs  
    ):  
        """  
        Initialize the medical QA specialized Qwen2 model.  
        """  
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
    
    def generate_medical_answer(  
        self,  
        query: str,  
        context: List[str],  
        max_length: int = 1024,  
        temperature: float = 0.5,  
        **kwargs  
    ) -> str:  
        """  
        Generate a medical answer based on the query and context.  
        
        Args:  
            query (str): The medical question  
            context (List[str]): The retrieved medical context passages  
            max_length (int): Maximum length of generated answer  
            temperature (float): Sampling temperature  
            **kwargs: Additional generation parameters  
            
        Returns:  
            str: The generated medical answer  
        """  
        inputs = self.prepare_inputs_for_rag(query, context)  
        
        if "messages" in inputs:  
            messages = inputs["messages"]  
            prompt = self.tokenizer.apply_chat_template(  
                messages,  
                return_tensors="pt",  
                add_generation_prompt=True  
            )  
        else:  
            prompt = inputs["prompt"]  
        
        return self.generate(  
            prompt=prompt,  
            max_length=max_length,  
            temperature=temperature,  
            **kwargs  
        )  
    
    def validate_medical_response(self, response: str, query: str, **kwargs) -> Dict[str, Any]:  
        """  
        Validate the medical response for accuracy and completeness.  
        
        Args:  
            response (str): The generated medical response  
            query (str): The original medical query  
            **kwargs: Additional validation parameters  
            
        Returns:  
            Dict[str, Any]: Validation results  
        """  
        # This is a placeholder for actual validation logic  
        # In a real implementation, this could check for medical accuracy,  
        # proper citations, completeness, etc.  
        validation_prompt = f"""  
        Evaluate the following medical response for accuracy, completeness, and helpfulness.  
        
        Query: {query}  
        
        Response: {response}  
        
        Evaluation:  
        """  
        
        evaluation = self.generate(validation_prompt, max_length=256, max_new_tokens=200, temperature=0.3)  
        
        return {  
            "is_valid": "accurate" in evaluation.lower() or "correct" in evaluation.lower(),  
            "evaluation": evaluation,  
            "response": response  
        }  
        
        
        
        



if __name__ == '__main__':
    
    '''
    python -m src.models.qwen_model
    '''
    
    from ..config.model_config import model_path
    model = Qwen2Model(model_path=model_path)
    
    
    result = model.generate("今天天气很不错，你觉得呢？")


    print(result)