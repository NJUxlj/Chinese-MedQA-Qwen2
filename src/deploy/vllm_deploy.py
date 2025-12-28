import os
import sys
import signal
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config.vllm_deployer_config import VLLMDeployerConfig


class VLLMDeployer:
    def __init__(self, config: VLLMDeployerConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.logger = setup_logger(name=__class__.__name__, level="INFO")
        self._validate_config()

    def _validate_config(self):
        import torch
        available_gpus = torch.cuda.device_count()
        if self.config.tensor_parallel_size > available_gpus:
            raise ValueError(
                f"请求的张量并行度 ({self.config.tensor_parallel_size}) "
                f"超过可用GPU数量 ({available_gpus})"
            )

    def _build_command(self) -> list:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name_or_path,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]

        if self.config.pipeline_parallel_size > 1:
            cmd.extend(["--pipeline-parallel-size", str(self.config.pipeline_parallel_size)])

        if self.config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])

        if self.config.dtype != "auto":
            cmd.extend(["--dtype", self.config.dtype])

        if self.config.quantization is not None:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.trust_remote_code is False:
            cmd.append("--trust-remote-code")

        if self.config.enforce_eager:
            cmd.append("--enforce-eager")

        if self.config.seed is not None and self.config.seed != 42:
            cmd.extend(["--seed", str(self.config.seed)])

        return cmd

    def deploy(self) -> subprocess.Popen:
        if self.process is not None:
            self.logger.warning("服务已启动，跳过重复部署")
            return self.process

        self.logger.info(f"启动 vLLM API 服务器...")
        self.logger.info(f"模型: {self.config.model_name_or_path}")
        self.logger.info(f"张量并行度: {self.config.tensor_parallel_size}")
        self.logger.info(f"GPU显存利用率: {self.config.gpu_memory_utilization}")

        cmd = self._build_command()
        self.logger.info(f"执行命令: {' '.join(cmd)}")

        env = os.environ.copy()
        cuda_devices = ",".join(str(i) for i in range(self.config.tensor_parallel_size))
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.logger.info(f"服务进程已启动，PID: {self.process.pid}")
        self._wait_for_server_ready(timeout=300)
        return self.process

    def _wait_for_server_ready(self, timeout: int = 300):
        url = "http://localhost:8000/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.logger.info("服务启动成功，API 服务器已就绪")
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        raise RuntimeError(f"服务启动超时（{timeout}秒）")

    def shutdown(self):
        if self.process is None:
            self.logger.warning("服务未启动，无需关闭")
            return

        self.logger.info(f"正在关闭服务 (PID: {self.process.pid})...")
        self.process.terminate()

        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.logger.warning("服务未响应，强制终止")
            self.process.kill()
            self.process.wait()

        self.process = None
        self.logger.info("服务已关闭")

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def health_check(self) -> Dict[str, Any]:
        status = {
            "is_running": self.is_running(),
            "model": self.config.model_name_or_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }

        if self.is_running():
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                status["api_healthy"] = response.status_code == 200
            except requests.exceptions.RequestException:
                status["api_healthy"] = False
        else:
            status["api_healthy"] = False

        return status

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return None

    def __enter__(self):
        self.deploy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        self.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="vLLM API 服务器部署工具")
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU显存利用率")
    parser.add_argument("--max-model-len", type=int, default=None, help="最大上下文长度")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    config = VLLMDeployerConfig(
        model_name_or_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    with VLLMDeployer(config) as deployer:
        deployer.logger.info("服务运行中，按 Ctrl+C 停止...")
        try:
            while deployer.is_running():
                time.sleep(10)
        except KeyboardInterrupt:
            deployer.logger.info("收到停止信号")


if __name__ == "__main__":
    main()
