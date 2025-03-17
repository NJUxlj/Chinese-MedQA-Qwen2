import streamlit as st
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

from config.config import MODEL_PATH

# 初始化模型（带缓存）
'''
@st.cache_resource 是 Streamlit 提供的一个装饰器，

它的主要作用是缓存资源密集型对象，
以避免在每次页面刷新或交互时重新创建这些对象。

使用 @st.cache_resource 的好处包括：

性能优化：模型只需加载一次，后续调用直接使用缓存结果
资源节省：避免重复占用显存和内存
'''
@st.cache_resource  
def load_model():  
    return ChatModel(  
        model_name_or_path=MODEL_PATH,  # 指向本地权重路径  
        template="chatml",  # Qwen2专用对话模板  
        trust_remote_code=True,  # 必需参数  
        load_in_4bit=True,  # 4bit量化节省显存  
        # 可选的设备映射配置  
        # device_map="auto"  
    )  

chat_model = load_model()
INIT_HISTORY = [
            (
            '现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。', 
            '当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。'
            )
        ]

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = INIT_HISTORY.copy()

# 页面配置
st.set_page_config(page_title="🧑⚕️ 中医助手")
st.title("🧑⚕️ 中医智能助手")
st.caption("输入您的中医问题，使用清除按钮重置对话")

# 聊天容器 【把历史记录写入容器内】
chat_container = st.container()
with chat_container:
    for query, response in st.session_state.history[1:]:  # 跳过初始提示 (第一轮对话)
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(response)

# 输入区域
if prompt := st.chat_input("请输入您的问题..."):
    # st.session_state.history.append({"role": "user", "content": prompt})  
    # 显示用户输入
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)
    
    # 生成响应
    with chat_container:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in chat_model.stream_chat(
                    prompt, 
                    st.session_state.history,
                    temperature=0.7,
                    repetition_penalty=1.1 
                ):
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
        
        
# streamlit run medical_chat.py  


'''
streamlit run medical_chat.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8080

'''