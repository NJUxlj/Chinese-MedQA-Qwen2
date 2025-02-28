import gradio as gr
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

# 初始化模型（仅在启动时加载一次）
chat_model = ChatModel()
INIT_HISTORY = [('现在你是一名专业的中医医生...（同原始初始化历史）')]

def clear_history():
    global chat_model
    torch_gc()
    return INIT_HISTORY.copy()

def respond(message, history):
    history = history or INIT_HISTORY.copy()
    
    # 流式生成响应
    full_response = ""
    for new_text in chat_model.stream_chat(message, history):
        full_response += new_text
        yield full_response
    
    # 更新历史（自动处理）

with gr.Blocks(title="中医聊天机器人") as demo:
    gr.Markdown("## 🧑⚕️ 中医智能助手")
    gr.Markdown("输入您的中医问题，使用下方按钮清除历史")
    
    chat = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.State(INIT_HISTORY.copy())
        ],
        retry_btn=None,
        undo_btn=None
    )
    
    with gr.Row():
        clear_btn = gr.Button("🧹 清除历史")
        clear_btn.click(
            fn=clear_history,
            outputs=chat.chatbot
        )

if __name__ == "__main__":
    demo.queue().launch()