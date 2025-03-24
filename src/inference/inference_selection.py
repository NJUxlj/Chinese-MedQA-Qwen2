import os
import logging
import time
import json
import yaml
import torch
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
from json import JSONDecodeError

from model.zhipu.zhipuAI import ZhipuAPIModel

# === 配置结构定义 ===
@dataclass
class InferenceConfig:
    model_path: str = "/models/qwen2-medical"
    inference_backend: str = "vllm"  # vllm|fastllm|zhipu
    max_seq_length: int = 4096
    temperature: float = 0.3
    top_p: float = 0.85
    api_timeout: int = 30
    tool_timeout: int = 10


# === 核心推理类 ===
class MedicalInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = self._init_engine()
        self.tools = self._load_tools()
        
    def _init_engine(self):
        """根据配置初始化推理引擎"""
        try:
            if self.config.inference_backend == "vllm":
                from vllm import LLM, SamplingParams
                self.sampling_params = SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_seq_length
                )
                return LLM(
                    model=self.config.model_path,
                    tensor_parallel_size=torch.cuda.device_count()
                )
                
            elif self.config.inference_backend == "fastllm":
                from fastllm import create_llm
                return create_llm(
                    model_path=self.config.model_path,
                    device_map="auto",
                    max_length=self.config.max_seq_length
                )
                
            elif self.config.inference_backend == "zhipu":
                return ZhipuAPIModel()
                
            else:
                raise ValueError(f"Unsupported backend: {self.config.inference_backend}")
        except ImportError as e:
            logging.error(f"依赖库未安装: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"引擎初始化失败: {str(e)}")
            raise

    # === 核心生成方法 ===
    @log_performance
    async def generate(self, prompt: str, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        统一生成接口
        :param prompt: 完整提示词（包含RAG上下文）
        :param tools: 可用工具列表（function calling）
        :return: 包含生成结果和元数据的字典
        """
        try:
            if tools or self._needs_tool_call(prompt):
                return await self._handle_function_call(prompt, tools or self.tools)
                
            if self.config.inference_backend == "vllm":
                return await self._generate_vllm(prompt)
                
            elif self.config.inference_backend == "fastllm":
                return await self._generate_fastllm(prompt)
                
            elif self.config.inference_backend == "zhipu":
                return await self._generate_zhipu(prompt)
                
        except Exception as e:
            logging.error(f"生成失败: {str(e)}")
            return {"error": str(e)}

    # === 各后端生成实现 ===
    async def _generate_vllm(self, prompt: str) -> Dict:
        from vllm import RequestOutput
        try:
            results_generator = self.engine.generate(
                prompt, 
                self.sampling_params,
                request_id=os.urandom(16).hex()
            )
            
            async for output in results_generator:
                if output.finished:
                    return {
                        "text": output.outputs[0].text,
                        "latency": output.latency,
                        "tokens": len(output.outputs[0].token_ids)
                    }
            return {"error": "生成未完成"}
        except Exception as e:
            logging.error(f"vLLM生成失败: {str(e)}")
            raise

    async def _generate_fastllm(self, prompt: str) -> Dict:
        try:
            result = await self.engine.async_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            return {
                "text": result,
                "tokens": len(self.engine.tokenizer.encode(result))
            }
        except Exception as e:
            logging.error(f"FastLLM生成失败: {str(e)}")
            raise

    async def _generate_zhipu(self, prompt: str) -> Dict:
        try:
            return await self.engine.generate(prompt)
        except Exception as e:
            logging.error(f"智谱API调用失败: {str(e)}")
            raise

    # === 工具调用处理 ===
    async def _handle_function_call(self, prompt: str, tools: List[Dict]) -> Dict:
        try:
            # 生成工具调用请求
            tool_call_prompt = self._build_tool_prompt(prompt, tools)
            raw_response = await self.generate(tool_call_prompt)
            if "error" in raw_response:
                return raw_response
                
            # 解析工具调用
            tool_call = self._parse_tool_call(raw_response["text"])
            if not tool_call:
                return {"error": "工具调用解析失败"}
                
            # 执行工具
            tool_result = await self._execute_tool(tool_call)
            if "error" in tool_result:
                return tool_result
                
            # 生成最终响应
            final_prompt = f"工具调用结果：{tool_result}\n请根据结果回答原问题：{prompt}"
            return await self.generate(final_prompt)
        except Exception as e:
            logging.error(f"工具调用流程失败: {str(e)}")
            return {"error": str(e)}

    def _parse_tool_call(self, response: str) -> Optional[Tuple[str, Dict]]:
        """解析工具调用响应（示例实现）"""
        try:
            func_start = response.find("<function>") + 10
            func_end = response.find("</function>")
            param_start = response.find("<params>") + 7
            param_end = response.find("</params>")
            
            tool_name = response[func_start:func_end].strip()
            params = json.loads(response[param_start:param_end].strip())
            return (tool_name, params)
        except (JSONDecodeError, ValueError) as e:
            logging.error(f"解析失败: {str(e)}")
            return None

    async def _execute_tool(self, tool_call: Tuple[str, Dict]) -> Dict:
        """执行工具调用（示例实现）"""
        tool_name, params = tool_call
        try:
            if tool_name == "drug_dose_calculator":
                return await self._calculate_drug_dose(**params)
            elif tool_name == "clinical_guideline_search":
                return await self._search_guidelines(**params)
            else:
                return {"error": "未知工具"}
        except Exception as e:
            logging.error(f"工具执行失败: {str(e)}")
            return {"error": str(e)}

    # === 示例工具实现 ===
    async def _calculate_drug_dose(self, age: int, weight: float, drug_name: str) -> Dict:
        """药物剂量计算示例"""
        # 这里应实现实际业务逻辑
        return {"dose": "100mg/day", "warning": "需监测肝功能"}

    async def _search_guidelines(self, keyword: str) -> Dict:
        """临床指南搜索示例"""
        # 这里应实现实际搜索逻辑
        return {"results": ["2023版糖尿病诊疗指南", "高血压管理指南"]}

    # === 辅助方法 ===
    def _needs_tool_call(self, prompt: str) -> bool:
        """简单判断是否需要工具调用"""
        return any(keyword in prompt for keyword in ["剂量", "指南", "计算"])

    def _build_tool_prompt(self, prompt: str, tools: List[Dict]) -> str:
        tool_desc = "\n".join([f"{t['name']}: {t['description']}" for t in tools])
        return f"""你是一个医疗助手，请根据可用工具选择最合适的工具调用：
        
                当前问题：{prompt}
                可用工具：
                {tool_desc}

                请严格使用以下XML格式响应：
                <function>工具名称</function>
                <params>JSON格式参数</params>"""

    def _load_tools(self) -> List[Dict]:
        """加载预定义工具列表"""
        return [{
            "name": "drug_dose_calculator",
            "description": "计算药物剂量，参数：age（年龄）, weight（体重kg）, drug_name（药品名）"
        }, {
            "name": "clinical_guideline_search",
            "description": "搜索临床指南，参数：keyword（关键词）"
        }]

    # === 配置管理 ===
    @classmethod
    def from_config_path(cls, config_path: str):
        """从YAML文件加载配置"""
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                config = InferenceConfig(**config_data)
                return cls(config)
        except Exception as e:
            logging.error(f"配置加载失败: {str(e)}")
            raise

    def reload_model(self, new_model_path: str):
        """热更新模型"""
        try:
            self.config.model_path = new_model_path
            self.engine = self._init_engine()
            logging.info("模型热更新成功")
        except Exception as e:
            logging.error(f"模型重载失败: {str(e)}")
            raise

# === 装饰器 ===
def log_performance(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start = time.time()
        try:
            result = await func(self, *args, **kwargs)
            latency = time.time() - start
            logging.info(f"生成耗时: {latency:.2f}s, 消耗token: {result.get('tokens', 0)}")
            return result
        except Exception as e:
            logging.error(f"执行失败: {str(e)}")
            return {"error": str(e)}
    return wrapper

# === 使用示例 ===
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例配置
    config = InferenceConfig(
        inference_backend="vllm",
        model_path="/path/to/your/model"
    )
    
    # 初始化推理引擎
    infer_engine = MedicalInference(config)
    
    # 测试普通查询
    async def test_query():
        response = await infer_engine.generate("如何诊断II型糖尿病？")
        print("测试普通查询:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    
    # 测试工具调用
    async def test_tool_call():
        response = await infer_engine.generate(
            "60岁患者体重75kg，使用二甲双胍的推荐剂量是多少？"
        )
        print("\n测试工具调用:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    
    # 运行测试
    import asyncio
    asyncio.run(test_query())
    asyncio.run(test_tool_call())
