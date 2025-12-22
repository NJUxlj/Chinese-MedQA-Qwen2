#!/usr/bin/env python3
"""
GPT-2模型转换为Ollama格式的转换器
将Hugging Face格式的GPT-2模型转换为Ollama兼容的格式
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional

def check_ollama_installed():
    """检查Ollama是否已安装"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama已安装: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama未正确安装")
            return False
    except FileNotFoundError:
        print("❌ Ollama未安装，请先安装Ollama")
        print("安装方法: https://ollama.ai")
        return False

def check_model_exists(model_path: str) -> bool:
    """检查模型路径是否存在"""
    path = Path(model_path)
    if path.exists():
        print(f"✅ 找到模型路径: {model_path}")
        
        # 检查必要的文件
        config_file = path / "config.json"
        model_files = list(path.glob("*.bin")) + list(path.glob("pytorch_model.bin*"))
        
        if config_file.exists() and model_files:
            print(f"✅ 模型文件完整，包含 {len(model_files)} 个模型文件")
            return True
        else:
            print("❌ 模型文件不完整，缺少config.json或权重文件")
            return False
    else:
        print(f"❌ 模型路径不存在: {model_path}")
        return False

def convert_to_gguf(model_path: str, output_path: str) -> bool:
    """
    将模型转换为GGUF格式
    需要使用transformers和gguf库
    """
    try:
        print("🔄 开始转换为GGUF格式...")
        
        # 创建转换脚本
        convert_script = f"""
import torch
from transformers import AutoModel, AutoTokenizer
from gguf import GGUF

# 加载模型和tokenizer
model_path = "{model_path}"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 保存为GGUF格式
model.save_pretrained("{output_path}")
tokenizer.save_pretrained("{output_path}")

print("✅ 模型转换完成")
"""
        
        # 保存转换脚本
        script_path = "/tmp/convert_gpt2.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(convert_script)
        
        # 运行转换
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd="/tmp")
        
        if result.returncode == 0:
            print("✅ GGUF转换成功")
            return True
        else:
            print(f"❌ GGUF转换失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 转换过程中出错: {str(e)}")
        return False

def create_ollama_modelfile(gpt2_path: str, output_name: str = "gpt2-custom") -> str:
    """创建Ollama Modelfile"""
    
    modelfile_content = f"""FROM {gpt2_path}

# GPT-2模型配置
TEMPLATE """ + "{{.Prompt}}" + """

# 参数设置
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 1024
PARAMETER num_predict 512

# 系统提示
SYSTEM """ + """您是一个有用的AI助手，能够回答各种问题并提供有用信息。""" + """
"""
    
    modelfile_path = f"/tmp/{output_name}_Modelfile"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✅ 创建Modelfile: {modelfile_path}")
    return modelfile_path

def create_ollama_model(modelfile_path: str, model_name: str) -> bool:
    """使用Modelfile创建Ollama模型"""
    try:
        print(f"🔄 创建Ollama模型: {model_name}")
        
        result = subprocess.run([
            'ollama', 'create', model_name, '-f', modelfile_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Ollama模型创建成功: {model_name}")
            return True
        else:
            print(f"❌ 模型创建失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 创建模型时出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== GPT-2 转 Ollama 模型转换器 ===\n")
    
    # 模型路径
    model_path = "models/gpt2"
    output_name = "gpt2-custom"
    
    # 1. 检查Ollama
    if not check_ollama_installed():
        return False
    
    # 2. 检查模型文件
    if not check_model_exists(model_path):
        return False
    
    # 3. 创建输出目录
    output_path = "/tmp/gpt2_gguf"
    os.makedirs(output_path, exist_ok=True)
    
    # 4. 转换格式（需要手动执行，因为依赖较多）
    print("\n⚠️  注意：自动转换需要安装额外的库")
    print("请手动执行以下步骤：")
    print(f"1. 安装依赖: pip install transformers accelerate gguf")
    print(f"2. 将{model_path}转换为GGUF格式")
    print(f"3. 使用转换后的模型创建Modelfile")
    
    # 5. 创建示例Modelfile
    modelfile_path = create_ollama_modelfile("/path/to/converted/model", output_name)
    
    print(f"\n📝 手动步骤:")
    print(f"1. 转换模型为GGUF格式")
    print(f"2. 修改modelfile中的FROM路径为转换后的模型路径")
    print(f"3. 运行: ollama create {output_name} -f {modelfile_path}")
    print(f"4. 测试模型: ollama run {output_name}")
    
    return True

if __name__ == "__main__":
    main()