import requests
import uuid

BASE_URL = "http://localhost:8000"

# 创建新会话
session_id = str(uuid.uuid4())
response = requests.post(f"{BASE_URL}/create_session")
session_id = response.json()["session_id"]

# 测试对话
def stream_chat(query: str):
    with requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "query": query},
        stream=True
    ) as r:
        for chunk in r.iter_content():
            print(chunk.decode("utf-8"), end="", flush=True)
    print()

while True:
    query = input("\n患者: ")
    if query.strip().lower() == "exit":
        break
    if query.strip().lower() == "clear":
        requests.post(f"{BASE_URL}/clear", json={"session_id": session_id})
        print("History cleared")
        continue
    print("医师: ", end="")
    stream_chat(query)