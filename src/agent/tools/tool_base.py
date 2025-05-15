# 基础工具类
import abc
from typing import Dict, Any, Optional, List

class ToolBase(abc.ABC):
    """工具基类，所有工具都应该继承这个类"""
    
    def __init__(
        self, 
        name: str,
        description: str,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
            parameters: 工具参数定义，格式为：
                {
                    "参数名": {
                        "type": "string|int|float|boolean|array|object",
                        "description": "参数描述",
                        "required": True|False,
                        "enum": ["可选值1", "可选值2"], # 可选
                        "default": "默认值" # 可选
                    }
                }
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    def _validate_parameters(self, **kwargs) -> List[str]:
        """
        验证参数是否符合定义
        
        Args:
            kwargs: 实际传入的参数
            
        Returns:
            错误消息列表，如果为空，则表示验证通过
        """
        errors = []
        
        # 检查必填参数
        for param_name, param_info in self.parameters.items():
            if param_info.get('required', False) and param_name not in kwargs:
                errors.append(f"缺少必填参数 '{param_name}'")
        
        # 检查参数类型
        for param_name, param_value in kwargs.items():
            if param_name not in self.parameters:
                errors.append(f"未知参数 '{param_name}'")
                continue
            
            param_info = self.parameters[param_name]
            param_type = param_info.get('type', 'any')
            
            # 类型检查
            if param_type == 'string' and not isinstance(param_value, str):
                errors.append(f"参数 '{param_name}' 类型错误，应为字符串")
            elif param_type == 'int' and not isinstance(param_value, int):
                errors.append(f"参数 '{param_name}' 类型错误，应为整数")
            elif param_type == 'float' and not isinstance(param_value, (int, float)):
                errors.append(f"参数 '{param_name}' 类型错误，应为浮点数")
            elif param_type == 'boolean' and not isinstance(param_value, bool):
                errors.append(f"参数 '{param_name}' 类型错误，应为布尔值")
            elif param_type == 'array' and not isinstance(param_value, list):
                errors.append(f"参数 '{param_name}' 类型错误，应为数组")
            elif param_type == 'object' and not isinstance(param_value, dict):
                errors.append(f"参数 '{param_name}' 类型错误，应为对象")
            
            # 枚举值检查
            if 'enum' in param_info and param_value not in param_info['enum']:
                enum_values = ', '.join([str(v) for v in param_info['enum']])
                errors.append(f"参数 '{param_name}' 的值 '{param_value}' 不在允许的范围 [{enum_values}] 内")
        
        return errors
    
    @abc.abstractmethod
    def _run(self, **kwargs) -> Any:
        """
        工具的具体实现，子类必须重写这个方法
        
        Args:
            kwargs: 参数
            
        Returns:
            工具执行结果
        """
        pass
    
    def run(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            kwargs: 参数
            
        Returns:
            工具执行结果
        """
        # 参数验证
        errors = self._validate_parameters(**kwargs)
        if errors:
            error_message = "\n".join(errors)
            raise ValueError(f"参数验证失败:\n{error_message}")
        
        # 添加默认值
        for param_name, param_info in self.parameters.items():
            if param_name not in kwargs and 'default' in param_info:
                kwargs[param_name] = param_info['default']
        
        # 执行工具
        return self._run(**kwargs)