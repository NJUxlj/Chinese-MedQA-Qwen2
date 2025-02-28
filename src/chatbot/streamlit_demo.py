import streamlit as st
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

# 初始化模型（带缓存）
@st.cache_resource
def load_model():
    return ChatModel()

chat_model = load_model()
INIT_HISTORY = [('现在你是一名专业的中医医生...（同原始初始化历史）')]

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = INIT_HISTORY.copy()

# 页面配置
st.set_page_config(page_title="🧑⚕️ 中医助手")
st.title("🧑⚕️ 中医智能助手")
st.caption("输入您的中医问题，使用清除按钮重置对话")

# 聊天容器
chat_container = st.container()
with chat_container:
    for query, response in st.session_state.history[1:]:  # 跳过初始提示
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(response)

# 输入区域
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户输入
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)
    
    # 生成响应
    with chat_container:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in chat_model.stream_chat(prompt, st.session_state.history):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
    
    # 更新历史记录
    st.session_state.history.append((prompt, full_response))

# 侧边栏控制
with st.sidebar:
    if st.button("🧹 清除历史"):
        st.session_state.history = INIT_HISTORY.copy()
        torch_gc()
        st.rerun()