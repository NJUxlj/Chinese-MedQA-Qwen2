import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config.settings import settings




class VLLMDeployer:
    def __init__(self, vllm_config=None) -> None:
        self.config = vllm_config if vllm_config is not None else settings.vllm
        self.process: Optional[subprocess.Popen] = None
        self.logger = setup_logger(name=__class__.__name__, level="INFO")
        self._validate_config()

    def _validate_config(self):
        try:
            import torch
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                self.logger.warning("未检测到 CUDA GPU，将在 CPU / MPS 模式下运行（vLLM 可能不支持）")
                return
            if self.config.tensor_parallel_size > available_gpus:
                raise ValueError(
                    f"请求的张量并行度 ({self.config.tensor_parallel_size}) "
                    f"超过可用GPU数量 ({available_gpus})"
                )
        except ImportError:
            self.logger.warning("torch 未安装，跳过 GPU 数量验证")

    def _get_base_url(self) -> str:
        return str(self.config.base_url)

    def _build_command(self) -> list:
        port = int(self.config.port)
        host = str(self.config.host)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.config.model_name_or_path),
            "--host", host,
            "--port", str(port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]

        if self.config.pipeline_parallel_size > 1:
            cmd.extend(["--pipeline-parallel-size", str(self.config.pipeline_parallel_size)])

        if self.config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])

        if self.config.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])

        if str(self.config.dtype) != "auto":
            cmd.extend(["--dtype", str(self.config.dtype)])

        if self.config.quantization is not None:
            cmd.extend(["--quantization", str(self.config.quantization)])

        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.config.enforce_eager:
            cmd.append("--enforce-eager")

        if self.config.seed is not None and int(self.config.seed) != 42:
            cmd.extend(["--seed", str(self.config.seed)])

        return cmd

    def deploy(self) -> subprocess.Popen:
        if self.process is not None:
            self.logger.warning("服务已启动，跳过重复部署")
            return self.process

        self.logger.info("启动 vLLM API 服务器...")
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
        health_url = f"{self._get_base_url()}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
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
        is_running = self.is_running()
        status = {
            "is_running": is_running,
            "model": str(self.config.model_name_or_path),
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }

        if is_running:
            health_url = f"{self._get_base_url()}/health"
            try:
                response = requests.get(health_url, timeout=5)
                status["api_healthy"] = response.status_code == 200
            except requests.exceptions.RequestException:
                status["api_healthy"] = False
        else:
            status["api_healthy"] = False

        return status

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        models_url = f"{self._get_base_url()}/v1/models"
        try:
            response = requests.get(models_url, timeout=5)
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="vLLM API 服务器部署工具")
    parser.add_argument("--model", type=str, default=None, help="模型路径或名称（默认读取config.yaml）")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU显存利用率")
    parser.add_argument("--max-model-len", type=int, default=None, help="最大上下文长度")
    args = parser.parse_args()

    vllm_cfg = settings.vllm
    if args.model:
        from omegaconf import OmegaConf
        vllm_cfg = OmegaConf.merge(vllm_cfg, {"model_name_or_path": args.model})
    if args.tensor_parallel_size is not None:
        from omegaconf import OmegaConf
        vllm_cfg = OmegaConf.merge(vllm_cfg, {"tensor_parallel_size": args.tensor_parallel_size})
    if args.gpu_memory_utilization is not None:
        from omegaconf import OmegaConf
        vllm_cfg = OmegaConf.merge(vllm_cfg, {"gpu_memory_utilization": args.gpu_memory_utilization})
    if args.max_model_len is not None:
        from omegaconf import OmegaConf
        vllm_cfg = OmegaConf.merge(vllm_cfg, {"max_model_len": args.max_model_len})

    deployer = VLLMDeployer(vllm_config=vllm_cfg)
    deployer.logger.info("服务运行中，按 Ctrl+C 停止...")
    deployer.deploy()
    try:
        while deployer.is_running():
            time.sleep(10)
    except KeyboardInterrupt:
        deployer.logger.info("收到停止信号")
    finally:
        deployer.shutdown()


if __name__ == "__main__":
    main()
