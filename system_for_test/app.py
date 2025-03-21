
import os
import sys
import argparse
from typing import Dict, List, Optional, Any
import yaml
import json
import time
import gradio as gr

from src.agent.agent import MedicalAgent
from src.utils.helpers import setup_logging, format_pubmed_results, save_conversation, load_conversation

# 设置日志
logger = setup_logging("./logs/app.log")

def load_agent(config_path: str) -> MedicalAgent:
    """
    加载医疗AI Agent。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        MedicalAgent实例
    """
    logger.info(f"正在从 {config_path} 加载配置")
    agent = MedicalAgent(config_path)
    logger.info("Agent加载成功")
    return agent

def process_query(agent, query, history, show_details=False):
    """
    处理用户查询并更新界面。
    
    Args:
        agent: MedicalAgent实例
        query: 用户查询
        history: 对话历史
        show_details: 是否显示详细处理信息
        
    Returns:
        更新后的对话历史和处理细节
    """
    logger.info(f"处理查询: {query}")
    
    # 调用Agent处理查询
    result = agent.process_query(query, use_tools=True)
    
    # 提取响应和处理细节
    response = result.get("response", "抱歉，处理查询时出错。")
    processing_time = result.get("processing_time", 0)
    
    # 更新历史
    history.append((query, response))
    
    # 生成详细信息
    details = ""
    if show_details:
        details += f"处理时间: {processing_time:.2f}秒\n\n"
        
        # 工具调用信息
        tool_calls = result.get("tool_call_results", [])
        if tool_calls:
            details += "### 工具调用\n\n"
            for i, tool_call in enumerate(tool_calls, 1):
                function_name = tool_call.get("function_name", "未知")
                tool_result = tool_call.get("result", {})
                
                details += f"{i}. 调用: {function_name}\n"
                
                if function_name == "search_pubmed":
                    articles = tool_result.get("results", [])
                    details += f"   找到 {len(articles)} 篇相关文章\n"
                    
                elif function_name == "search_local_knowledge":
                    docs = tool_result.get("results", [])
                    details += f"   找到 {len(docs)} 个本地知识库匹配项\n"
                    
                elif function_name == "fetch_article_details":
                    article = tool_result.get("article")
                    if article:
                        details += f"   文章 {article.get('pmid')} - {article.get('title')}\n"
                    else:
                        details += f"   未找到文章\n"
                        
                details += "\n"
        
        # 检索上下文信息
        contexts = result.get("retrieved_contexts", [])
        if contexts:
            details += "### 检索的上下文\n\n"
            for i, context in enumerate(contexts, 1):
                # 截断长上下文
                if len(context) > 300:
                    context = context[:300] + "...(已截断)"
                details += f"{i}. {context}\n\n"
                
        # 显示初始和最终响应的差异
        initial_response = result.get("initial_response", "")
        if initial_response and initial_response != response:
            details += "### 响应改进\n\n"
            details += f"初始响应:\n{initial_response[:300]}...(已截断)\n\n"
            details += f"经反思后的响应:\n{response[:300]}...(已截断)\n\n"
    
    return history, details

def create_ui(agent):
    """
    创建Gradio Web界面。
    
    Args:
        agent: MedicalAgent实例
        
    Returns:
        Gradio界面实例
    """
    with gr.Blocks(title="医疗AI Agent") as demo:
        gr.Markdown("# 医疗AI Agent\n基于大模型的医疗信息助手，具备PubMed和本地知识库的检索能力")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600)
                msg = gr.Textbox(
                    placeholder="请输入您的医疗问题，例如：'Omicron变种的主要症状有哪些？'",
                    lines=3
                )
                with gr.Row():
                    submit_btn = gr.Button("提交")
                    clear_btn = gr.Button("清除对话")
                
                show_details_checkbox = gr.Checkbox(label="显示处理细节", value=False)
                
            with gr.Column(scale=2):
                details_md = gr.Markdown(label="处理细节")
                
        # 本地知识库管理
        with gr.Accordion("知识库管理", open=False):
            index_btn = gr.Button("索引本地文档")
            index_status = gr.Textbox(label="索引状态", interactive=False)
                
        # 事件处理
        submit_btn.click(
            process_query, 
            inputs=[agent, msg, chatbot, show_details_checkbox], 
            outputs=[chatbot, details_md],
            api_name="process_query"
        )
        
        msg.submit(
            process_query, 
            inputs=[agent, msg, chatbot, show_details_checkbox], 
            outputs=[chatbot, details_md]
        )
        
        clear_btn.click(
            lambda: ([], ""), 
            outputs=[chatbot, details_md],
            api_name="clear"
        )
        
        # 知识库索引功能
        def index_local_docs(agent):
            try:
                agent.index_local_documents()
                return "本地知识库索引完成"
            except Exception as e:
                return f"索引过程中出错: {str(e)}"
                
        index_btn.click(
            index_local_docs,
            inputs=[agent],
            outputs=[index_status],
            api_name="index_docs"
        )
        
    return demo

def main():
    parser = argparse.ArgumentParser(description="医疗AI Agent")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="配置文件路径")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 加载Agent
    agent = load_agent(args.config)
    
    # 启动UI
    demo = create_ui(agent)
    demo.launch(server_name="0.0.0.0", server_port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
