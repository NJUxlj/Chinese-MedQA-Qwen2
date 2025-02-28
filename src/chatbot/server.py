from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
from contextlib import asynccontextmanager
import uuid

# 全局状态管理
class ChatState:
    def __init__(self):
        self.sessions = {}
        self.model = None

    def init_model(self):
        self.model = ChatModel()
        
    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [
                    ('现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。', 
                     '当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。'
                    )
                ]
            }
        return self.sessions[session_id]

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_state.init_model()
    yield
    torch_gc()  # 服务关闭时清理内存

app = FastAPI(lifespan=lifespan)
chat_state = ChatState()

# API端点
@app.post("/chat")
async def chat_endpoint(session_id: str, query: str):
    session = chat_state.get_session(session_id)
    
    def generate_stream():
        full_response = ""
        for chunk in chat_state.model.stream_chat(query, session["history"]):
            full_response += chunk
            yield chunk.encode("utf-8")
        session["history"].append((query, full_response))
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

@app.post("/clear")
async def clear_history(session_id: str):
    if session_id not in chat_state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_state.sessions[session_id]["history"] = [
        ('现在你是一名专业的中医医生...', '当然，我将尽力为您提供...')
    ]
    torch_gc()
    return {"status": "History cleared"}

@app.post("/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    chat_state.get_session(session_id)  # 初始化新会话
    return {"session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)