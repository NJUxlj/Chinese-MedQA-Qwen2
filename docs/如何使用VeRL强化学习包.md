- 确保创建一个新的conda环境来运行verl项目

```bash
 conda config --add envs_dirs /root/autodl-tmp/envs    # 新添加一个环境存方法目录，防止系统盘存储空间不足

 conda create --name verl python=3.11

 conda activate verl
```

```bash
pip install torch==2.4    #最稳定的版本

cd path/to/Chinese-MedQA-Qwen2/verl

pip install -r requirements.txt

pip install --no-deps -e .

```