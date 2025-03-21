
from src.llm.qwen_model import QwenLLM
from src.llm.zhipu_model import ZhipuLLM

def get_llm(config):
    """
    根据配置获取适当的LLM实例。
    
    Args:
        config: 配置字典，包含LLM提供商和相关配置。
        
    Returns:
        LLM实例。
    """
    provider = config['llm']['provider'].lower()
    
    if provider == 'qwen2':
        return QwenLLM(config['llm']['qwen'])
    elif provider == 'zhipu':
        return ZhipuLLM(config['llm']['zhipu'])
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}，请使用 'qwen2' 或 'zhipu'")
