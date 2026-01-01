"""
MDAgents Gradio UI
医疗多智能体系统的用户界面
"""

import gradio as gr
import json
import asyncio
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from gradio import components as grc
from gradio.components import *


API_BASE_URL = "http://localhost:8000/api/mdagents"


class MDAgentsUI:
    """MDAgents用户界面类"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        """初始化UI"""
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """加载示例查询"""
        return [
            {
                "label": "简单感冒咨询",
                "query": "患者最近两天出现发热、咳嗽、流鼻涕症状，体温最高38.5°C",
                "context": {"age": 35, "gender": "male"}
            },
            {
                "label": "慢性病管理",
                "query": "患者男，65岁，有高血压和糖尿病史，近期出现胸痛和呼吸困难",
                "context": {"age": 65, "gender": "male", "history": ["高血压", "糖尿病"]}
            },
            {
                "label": "复杂多学科会诊",
                "query": "患者女，58岁，有冠心病、糖尿病、慢性肾病史，近期出现下肢水肿、夜间呼吸困难、心电图显示ST段压低，肾功能检查显示eGFR 45ml/min",
                "context": {"age": 58, "gender": "female", "history": ["冠心病", "糖尿病", "慢性肾病"]}
            },
            {
                "label": "术后康复咨询",
                "query": "患者3天前进行了腹腔镜胆囊切除手术，术后第一天情况良好，今天出现发热38.8°C、右上腹疼痛加剧，WBC 15.5×10^9/L",
                "context": {"age": 45, "gender": "female", "surgery": "腹腔镜胆囊切除术"}
            }
        ]
    
    async def call_api(self, endpoint: str, data: Dict) -> Dict:
        """调用API"""
        try:
            url = f"{self.api_base_url}/{endpoint}"
            response = await self.client.post(url, json=data, timeout=120.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": f"API请求失败: {str(e)}"}
        except Exception as e:
            return {"error": f"发生错误: {str(e)}"}
    
    def format_complexity(self, analysis: Dict) -> str:
        """格式化复杂度分析结果"""
        if not analysis:
            return "暂无分析结果"
        
        level = analysis.get("complexity_level", "unknown")
        score = analysis.get("complexity_score", 0)
        confidence = analysis.get("confidence", 0)
        
        level_colors = {
            "simple": "🟢",
            "moderate": "🟡",
            "complex": "🔴"
        }
        
        level_text = {
            "simple": "简单",
            "moderate": "中等",
            "complex": "复杂"
        }
        
        emoji = level_colors.get(level, "⚪")
        text = level_text.get(level, level)
        
        features = analysis.get("features", {})
        if features:
            features_text = f"""
### 详细特征
- **文本长度**: {features.get('text_length', 0)} 字符
- **症状数量**: {features.get('symptom_count', 0)}
- **医疗术语数**: {features.get('medical_terms_count', 0)}
- **涉及专科数**: {features.get('specialty_count', 0)}
- **诊断复杂度**: {features.get('diagnostic_complexity', 0):.2f}
- **治疗复杂度**: {features.get('treatment_complexity', 0):.2f}
- **急症关键词**: {', '.join(features.get('emergency_keywords', [])) or '无'}
"""
        else:
            features_text = ""
        
        recommended = analysis.get("recommended_agents", [])
        
        return f"""## 复杂度分析结果

**复杂度等级**: {emoji} {text} ({score:.2f})

**置信度**: {confidence:.2f}

**推荐智能体**: {', '.join(recommended) if recommended else '无'}

{features_text}"""
    
    def format_strategy(self, strategy: Dict) -> str:
        """格式化协作策略结果"""
        if not strategy:
            return "暂无协作策略"
        
        name = strategy.get("name", "未知")
        level = strategy.get("complexity_level", "unknown")
        agents = strategy.get("agents_needed", [])
        time_est = strategy.get("estimated_time", 0)
        confidence = strategy.get("confidence", 0)
        rules = strategy.get("adaptive_rules", [])
        fallbacks = strategy.get("fallback_strategies", [])
        
        return f"""## 协作策略

**策略名称**: {name}

**复杂度等级**: {level}

**参与智能体**: 
{chr(10).join(['- ' + a for a in agents]) if agents else '- 无'}

**预计时间**: {time_est} 秒

**置信度**: {confidence:.2f}

**自适应规则**:
{chr(10).join(['- ' + r for r in rules]) if rules else '- 无'}

**备用策略**:
{chr(10).join(['- ' + f for f in fallbacks]) if fallbacks else '- 无'}"""
    
    def format_agent_results(self, results: List[Dict]) -> str:
        """格式化智能体结果"""
        if not results:
            return "暂无智能体结果"
        
        formatted = ["## 智能体分析结果\n"]
        
        for i, result in enumerate(results, 1):
            agent_type = result.get("agent_type", "未知")
            agent_name = result.get("agent_name", "未知")
            status = result.get("status", "未知")
            confidence = result.get("confidence_score", 0)
            diagnosis = result.get("diagnosis", [])
            treatment = result.get("treatment_plan", "")
            reasoning = result.get("reasoning", "")
            
            status_emoji = {
                "completed": "✅",
                "running": "🔄",
                "pending": "⏳",
                "error": "❌"
            }.get(status, "⚪")
            
            formatted.append(f"""### {i}. {agent_name} ({agent_type})
**状态**: {status_emoji} {status}
**置信度**: {confidence:.2f}

**诊断结果**:
{chr(10).join(['- ' + d for d in diagnosis]) if diagnosis else '- 无'}

**治疗方案**:
{treatment if treatment else '暂无'}

**推理过程**:
{reasoning if reasoning else '暂无'}

---
""")
        
        return "\n".join(formatted)
    
    def format_consensus(self, consensus: Dict) -> str:
        """格式化共识结果"""
        if not consensus:
            return "暂无共识结果"
        
        level = consensus.get("consensus_level", 0)
        confidence = consensus.get("confidence_score", 0)
        method = consensus.get("method_used", "未知")
        diagnosis = consensus.get("final_diagnosis", {})
        treatment = consensus.get("final_treatment", {})
        supporters = consensus.get("supporting_agents", [])
        
        progress = "█" * int(level * 10) + "░" * (10 - int(level * 10))
        
        return f"""## 共识结果

**共识水平**: [{progress}] {level:.2f}

**置信度**: {confidence:.2f}

**共识方法**: {method}

**最终诊断**:
{json.dumps(diagnosis, ensure_ascii=False, indent=2) if diagnosis else '暂无'}

**最终治疗方案**:
{json.dumps(treatment, ensure_ascii=False, indent=2) if treatment else '暂无'}

**支持智能体**: {', '.join(supporters) if supporters else '无'}"""
    
    def format_final_report(self, report: Dict) -> str:
        """格式化最终报告"""
        if not report:
            return "暂无最终报告"
        
        return f"""## 最终报告

```json
{json.dumps(report, ensure_ascii=False, indent=2)}
```"""
    
    async def analyze_query(
        self,
        query: str,
        age: str,
        gender: str,
        history: str,
        options: Dict
    ) -> tuple:
        """分析查询"""
        if not query or not query.strip():
            return (
                "⚠️ 请输入医疗问题描述",
                "",
                "",
                "",
                "",
                "",
                "",
                0
            )
        
        patient_context = {}
        if age:
            try:
                patient_context["age"] = int(age)
            except ValueError:
                pass
        if gender:
            patient_context["gender"] = gender
        if history:
            patient_context["history"] = [h.strip() for h in history.split(",") if h.strip()]
        
        show_steps = options.get("show_steps", True) if options else True
        
        try:
            response = await self.call_api("query", {
                "query": query,
                "patient_context": patient_context if patient_context else None,
                "options": {"show_steps": show_steps}
            })
            
            if "error" in response:
                return (
                    f"❌ {response['error']}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    0
                )
            
            session_id = response.get("session_id", "未知")
            processing_time = response.get("processing_time", 0)
            timestamp = response.get("timestamp", "")
            
            complexity = response.get("complexity_analysis", {})
            strategy = response.get("collaboration_strategy", {})
            agents = response.get("agent_results", [])
            consensus = response.get("consensus_result", {})
            report = response.get("final_report", {})
            
            complexity_text = self.format_complexity(complexity)
            strategy_text = self.format_strategy(strategy)
            agents_text = self.format_agent_results(agents)
            consensus_text = self.format_consensus(consensus)
            report_text = self.format_final_report(report)
            
            full_result = f"""
# 医疗智能体会诊报告

**会话ID**: {session_id}
**处理时间**: {processing_time:.2f} 秒
**时间戳**: {timestamp}

---

{complexity_text}

---

{strategy_text}

---

{agents_text}

---

{consensus_text}

---

{report_text}
"""
            
            return (
                complexity_text,
                strategy_text,
                agents_text,
                consensus_text,
                report_text,
                full_result,
                processing_time
            )
            
        except Exception as e:
            return (
                f"❌ 处理出错: {str(e)}",
                "",
                "",
                "",
                "",
                "",
                0
            )
    
    async def quick_analyze(self, query: str) -> str:
        """快速分析查询复杂度"""
        if not query or not query.strip():
            return "⚠️ 请输入医疗问题描述"
        
        try:
            response = await self.call_api("complexity/analyze", {
                "query": query,
                "include_features": True
            })
            
            if "error" in response:
                return f"❌ {response['error']}"
            
            return self.format_complexity(response)
            
        except Exception as e:
            return f"❌ 分析失败: {str(e)}"
    
    async def get_system_status(self) -> str:
        """获取系统状态"""
        try:
            response = await self.call_api("status", {})
            
            if "error" in response:
                return f"❌ {response['error']}"
            
            status = response.get("status", "未知")
            total_agents = response.get("total_agents", 0)
            active_agents = response.get("active_agents", 0)
            consensus_rate = response.get("consensus_rate", 0)
            uptime = response.get("uptime", "未知")
            
            return f"""## 系统状态

**运行状态**: {'🟢 在线' if status == 'running' else '🔴 离线'}

**智能体统计**:
- 总智能体数: {total_agents}
- 活跃智能体: {active_agents}

**共识成功率**: {consensus_rate:.2%}

**运行时间**: {uptime}"""
            
        except Exception as e:
            return f"❌ 获取状态失败: {str(e)}"
    
    def load_example(self, example_index: int) -> tuple:
        """加载示例"""
        if example_index < 0 or example_index >= len(self.examples):
            return "", "", ""
        
        example = self.examples[example_index]
        query = example.get("query", "")
        context = example.get("context", {})
        
        age = str(context.get("age", ""))
        gender = context.get("gender", "")
        history = ", ".join(context.get("history", []))
        
        return query, age, history
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        with gr.Blocks(
            title="🏥 医疗多智能体会诊系统"
        ) as demo:
            
            header = gr.Markdown("""
            <div class="main-header">
                <h1>🏥 医疗多智能体会诊系统</h1>
                <p>基于人工智能的多学科医疗诊断辅助系统</p>
            </div>
            """)
            
            with gr.Tabs():
                with gr.TabItem("📋 智能会诊", id="consultation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 患者信息")
                            
                            query_input = gr.Textbox(
                                label="🗣️ 症状描述",
                                placeholder="请详细描述患者的症状、持续时间、既往史等信息...",
                                lines=5,
                                interactive=True
                            )
                            
                            with gr.Row():
                                age_input = gr.Textbox(
                                    label="👤 年龄",
                                    placeholder="例如: 65",
                                    scale=1
                                )
                                gender_input = gr.Dropdown(
                                    label="性别",
                                    choices=["male", "female"],
                                    value="male",
                                    scale=1
                                )
                            
                            history_input = gr.Textbox(
                                label="📖 既往史",
                                placeholder="用逗号分隔，例如: 高血压, 糖尿病, 冠心病",
                                lines=2
                            )
                            
                            with gr.Accordion("⚙️ 高级选项", open=False):
                                show_steps = gr.Checkbox(
                                    label="显示处理步骤",
                                    value=True
                                )
                            
                            with gr.Row():
                                submit_btn = gr.Button(
                                    "🚀 开始会诊",
                                    variant="primary",
                                    size="lg"
                                )
                                clear_btn = gr.Button(
                                    "🗑️ 清空",
                                    variant="secondary"
                                )
                            
                            gr.Markdown("### 💡 示例查询")
                            example_buttons = []
                            for i, example in enumerate(self.examples):
                                example_buttons.append(
                                    gr.Button(
                                        f"{example['label']}",
                                        variant="secondary",
                                        size="small"
                                    )
                                )
                            
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.TabItem("📊 复杂度分析", id="complexity"):
                                    complexity_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                                
                                with gr.TabItem("🤝 协作策略", id="strategy"):
                                    strategy_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                                
                                with gr.TabItem("👨‍⚕️ 智能体分析", id="agents"):
                                    agents_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                                
                                with gr.TabItem("✅ 共识结果", id="consensus"):
                                    consensus_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                                
                                with gr.TabItem("📝 最终报告", id="report"):
                                    report_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                                
                                with gr.TabItem("📄 完整报告", id="full"):
                                    full_output = gr.Markdown(
                                        value="等待输入...",
                                        elem_classes=["result-card"]
                                    )
                            
                            with gr.Row():
                                time_output = gr.Number(
                                    label="⏱️ 处理时间 (秒)",
                                    value=0,
                                    interactive=False
                                )
                                status_output = gr.Textbox(
                                    label="📍 状态",
                                    value="等待中",
                                    interactive=False
                                )
                    
                    def on_example_click(example_idx):
                        return lambda: self.load_example(example_idx)
                    
                    for i, btn in enumerate(example_buttons):
                        btn.click(
                            on_example_click(i),
                            outputs=[query_input, age_input, history_input]
                        )
                    
                    clear_btn.click(
                        lambda: ("", "", "", "", gr.update(value=True)),
                        outputs=[query_input, age_input, history_input, show_steps]
                    )
                    
                    submit_event = submit_btn.click(
                        self.analyze_query,
                        inputs=[query_input, age_input, gender_input, history_input, 
                               gr.State({"show_steps": True})],
                        outputs=[
                            complexity_output,
                            strategy_output,
                            agents_output,
                            consensus_output,
                            report_output,
                            full_output,
                            time_output,
                            status_output
                        ]
                    )
                
                with gr.TabItem("⚡ 快速分析", id="quick"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ### 快速复杂度分析
                            快速评估医疗查询的复杂度等级和推荐智能体
                            """)
                            
                            quick_query = gr.Textbox(
                                label="输入医疗查询",
                                placeholder="输入要分析的医疗问题...",
                                lines=3
                            )
                            
                            quick_analyze_btn = gr.Button(
                                "🔍 分析复杂度",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=2):
                            quick_output = gr.Markdown(value="等待输入...")
                    
                    quick_analyze_btn.click(
                        self.quick_analyze,
                        inputs=[quick_query],
                        outputs=[quick_output]
                    )
                
                with gr.TabItem("🔧 系统状态", id="status"):
                    with gr.Row():
                        refresh_status_btn = gr.Button(
                            "🔄 刷新状态",
                            variant="secondary"
                        )
                    
                    status_output = gr.Markdown(value="点击刷新获取系统状态...")
                    
                    refresh_status_btn.click(
                        self.get_system_status,
                        outputs=[status_output]
                    )
                
                with gr.TabItem("ℹ️ 使用说明", id="help"):
                    gr.Markdown("""
                    # 🏥 医疗多智能体会诊系统使用指南

                    ## 系统概述
                    本系统是一个基于人工智能的医疗多学科会诊辅助系统，通过多个专业智能体协同工作，为用户提供全面的医疗分析和建议。

                    ## 智能体介绍

                    ### 👨‍⚕️ 初级诊疗医生 (PCC)
                    负责初步症状分析和基础诊断

                    ### 🏥 专科医生 (Specialist)
                    根据具体病情，调用相应的专科知识进行深入分析

                    ### 👨‍💼 协调员 (Moderator)
                    协调各智能体的工作流程，确保诊断过程有序进行

                    ### 👥 招募者 (Recruiter)
                    根据病情复杂程度，决定需要哪些智能体参与

                    ### 📝 审查员 (Reviewer)
                    对最终诊断和治疗方案进行质量审查

                    ### 🔗 整合者 (Integrator)
                    整合各智能体的意见，形成最终的综合报告

                    ## 使用流程

                    1. **输入症状描述**：详细描述患者的症状、持续时间等信息
                    2. **提供患者信息**：包括年龄、性别、既往史等
                    3. **提交会诊**：点击"开始会诊"按钮
                    4. **查看结果**：系统将展示复杂度分析、协作策略、智能体分析、共识结果和最终报告

                    ## 复杂度等级

                    - 🟢 **简单**：基础症状描述，通常由单一智能体处理
                    - 🟡 **中等**：需要多个智能体协作处理
                    - 🔴 **复杂**：需要多学科会诊，所有相关智能体参与

                    ## 注意事项

                    ⚠️ **免责声明**：
                    - 本系统仅供参考，不能替代专业医疗诊断
                    - 最终诊断和治疗方案请咨询专业医生
                    - 系统输出不构成医学建议
                    """)
            
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #64748b;">
                <p>🏥 医疗多智能体会诊系统 | Powered by AI</p>
                <p>⚠️ 本系统仅供参考，不能替代专业医疗诊断</p>
            </div>
            """)
        
        return demo
    
    async def close(self):
        """关闭异步客户端"""
        await self.client.aclose()


async def main():
        """主函数"""
        ui = MDAgentsUI()
        demo = ui.create_interface()
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
