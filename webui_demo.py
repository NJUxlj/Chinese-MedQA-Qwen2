# from llmtuner import ChatModel
# from llmtuner.extras.misc import torch_gc

import gradio as gr


try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError: 
    print("Install `readline` for a better experience.")






# 输入name字符串，输出Hello {name}!字符串
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
    allow_flagging="never",
)



if __name__ == "__main__":
    demo.launch()





def main():
    pass







