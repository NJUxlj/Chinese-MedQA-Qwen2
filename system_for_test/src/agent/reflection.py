
from typing import Dict, List, Optional, Any, Tuple
import re

class ReflectionEngine:
    """
    反思引擎，用于改进Agent的响应质量，实现自校正和响应完善
    """
    
    def __init__(self, llm, config: Dict[str, Any]):
        """
        初始化反思引擎。
        
        Args:
            llm: 语言模型实例
            config: 反思引擎配置
        """
        self.llm = llm
        self.max_reflection_rounds = config.get('max_reflection_rounds', 2)
        self.reflection_strategy = config.get('reflection_strategy', 'cot')
        self.temperature = config.get('temperature', 0.1)
        
    def reflect_on_response(
        self, 
        query: str, 
        initial_response: str, 
        retrieved_context: Optional[List[str]] = None
    ) -> str:
        """
        对初始响应进行反思并改进。
        
        Args:
            query: 用户的原始查询
            initial_response: 模型的初始响应
            retrieved_context: 检索到的上下文信息（可选）
            
        Returns:
            反思后的改进响应
        """
        # 准备上下文信息
        context_str = ""
        if retrieved_context and len(retrieved_context) > 0:
            context_str = "以下是相关的参考信息:\n" + "\n---\n".join(retrieved_context)
        
        # 选择反思策略
        if self.reflection_strategy == 'cot':
            return self._chain_of_thought_reflection(query, initial_response, context_str)
        elif self.reflection_strategy == 'react':
            return self._react_reflection(query, initial_response, context_str)
        else:
            return self._default_reflection(query, initial_response, context_str)
            
    def _default_reflection(self, query: str, initial_response: str, context_str: str) -> str:
        """
        默认的反思策略，简单评估并改进响应。
        """
        system_prompt = """你是一位专业的医学AI助手，正在评估一个医疗问题的回答。
请分析给定的问题和初始回答，考虑以下几点：
1. 回答是否准确、全面地解答了问题
2. 回答是否基于科学依据和最新医学知识
3. 回答是否清晰、结构良好、易于理解
4. 回答是否遗漏了重要信息或存在错误

请提供一个改进后的回答，确保内容更准确、全面和有帮助。如果初始回答已经很好，可以保留它。"""

        reflection_prompt = f"""问题: {query}

初始回答:
{initial_response}

{context_str if context_str else ""}

请分析上述回答的质量，并提供一个改进后的回答。"""

        # 生成反思
        improved_response = self.llm.generate(
            prompt=reflection_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature
        )
        
        # 提取改进后的回答(移除分析部分)
        if "改进后的回答:" in improved_response:
            improved_response = improved_response.split("改进后的回答:")[1].strip()
        
        return improved_response
        
    def _chain_of_thought_reflection(self, query: str, initial_response: str, context_str: str) -> str:
        """
        使用链式思考(Chain of Thought)进行多步骤反思
        """
        system_prompt = """你是一位专业的医学AI助手，使用链式思考法来改进医疗问题的回答。请按照以下步骤进行：

第一步: 分析问题，确定需要回答的核心医学问题和相关领域。
第二步: 评估初始回答的优点和不足，特别关注医学准确性、全面性和有用性。
第三步: 结合参考资料（如果有），识别初始回答中的错误、误导或遗漏的信息。
第四步: 重新构建一个全面、准确且有针对性的回答。
第五步: 最终检查确保回答专业、平衡且符合最新医学共识。

在做出改进后的回答时，请确保：
- 使用准确的医学术语
- 提供科学依据支持关键观点
- 表明信息的确定性程度（确定、可能、尚无定论等）
- 结构清晰，易于普通人理解
- 在适当情况下提及可能的替代观点或方法"""

        reflection_prompt = f"""问题: {query}

初始回答:
{initial_response}

{context_str if context_str else ""}

请使用链式思考法对上述回答进行反思并提供改进。在完成所有分析步骤后，请给出最终改进的完整回答。"""

        # 进行多轮反思
        current_response = initial_response
        for i in range(self.max_reflection_rounds):
            improved_response = self.llm.generate(
                prompt=reflection_prompt.replace(initial_response, current_response),
                system_prompt=system_prompt,
                temperature=self.temperature
            )
            
            # 提取最终改进的回答
            final_answer_match = re.search(r"(最终回答|改进后的回答|完整回答)[:：]([\s\S]+)", improved_response)
            if final_answer_match:
                current_response = final_answer_match.group(2).strip()
            else:
                # 尝试从最后一段提取
                paragraphs = improved_response.split("\n\n")
                if len(paragraphs) > 0:
                    current_response = paragraphs[-1].strip()
                else:
                    current_response = improved_response
        
        return current_response
        
    def _react_reflection(self, query: str, initial_response: str, context_str: str) -> str:
        """
        使用ReAct方法(Reasoning and Acting)进行反思，适合需要推理和行动的问题
        """
        system_prompt = """你是一位专业的医学AI助手，使用"推理和行动"(ReAct)方法来回答医疗问题。
请遵循以下流程：

思考(Thought): 分析问题需要什么医学知识和信息。思考初始回答的优缺点。
行动(Action): 确定需要改进的方向，包括添加遗漏的信息、纠正错误、改进结构等。
观察(Observation): 评估改进方向的合理性，参考提供的医学信息。
最终回答(Answer): 基于以上推理过程，提供最终改进的医学回答。

请确保你的回答具有很高的医学准确性，清晰解释复杂概念，并在必要时提及信息的置信度水平。"""

        reflection_prompt = f"""问题: {query}

初始回答:
{initial_response}

{context_str if context_str else ""}

请使用ReAct方法（思考-行动-观察-回答）对上述回答进行改进。完成所有步骤后，给出一个最终的全面准确回答。"""

        improved_response = self.llm.generate(
            prompt=reflection_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature
        )
        
        # 提取最终回答
        final_answer_match = re.search(r"(最终回答|Answer)[:：]([\s\S]+)", improved_response)
        if final_answer_match:
            return final_answer_match.group(2).strip()
        else:
            # 如果没有找到明确标记，返回改进后的完整回答
            return improved_response
