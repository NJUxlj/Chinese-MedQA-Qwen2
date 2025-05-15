from typing import Dict, List, Optional, Any, Set

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ToolManager:
    """
    工具管理类，负责管理和组织Agent可用的工具
    """
    
    def __init__(self) -> None:
        """初始化工具管理器"""
        self.tools = {}  # 工具名称 -> 工具实例
    
    def add_tool(self, tool) -> None:
        """
        添加工具
        
        Args:
            tool: 工具实例
        """
        tool_name = tool.name
        if tool_name in self.tools:
            logger.warning(f"覆盖已存在的工具 '{tool_name}'")
        
        self.tools[tool_name] = tool
        logger.info(f"添加工具 '{tool_name}'")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否成功移除
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"移除工具 '{tool_name}'")
            return True
        
        logger.warning(f"尝试移除不存在的工具 '{tool_name}'")
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        获取工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例，如果不存在则返回None
        """
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, Any]:
        """
        获取所有工具
        
        Returns:
            工具字典 {工具名称 -> 工具实例}
        """
        return self.tools
    
    def has_tools(self) -> bool:
        """
        检查是否有工具
        
        Returns:
            是否有工具
        """
        return len(self.tools) > 0
    
    def get_tools_description(self) -> str:
        """
        获取所有工具的描述
        
        Returns:
            所有工具的描述，格式化为文本
        """
        if not self.tools:
            return "无可用工具"
        
        descriptions = []
        for name, tool in self.tools.items():
            param_desc = ""
            if hasattr(tool, 'parameters'):
                params = tool.parameters
                param_list = []
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'any')
                    param_desc_text = param_info.get('description', '')
                    required = param_info.get('required', False)
                    param_list.append(f"- {param_name} ({param_type}{'，必填' if required else ''}): {param_desc_text}")
                
                if param_list:
                    param_desc = "\n参数:\n" + "\n".join(param_list)
            
            descriptions.append(f"工具名称: {name}\n描述: {tool.description}{param_desc}\n")
        
        return "\n".join(descriptions)