'''
评估模型在代码测试集上的性能
'''
import os
import sys
import time
import subprocess
import tempfile
import signal
import json
import re
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from threading import Lock
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.base_evaluator import BaseEvaluator, EvaluatorDataset


class CodeSecurityChecker:
    DANGEROUS_PATTERNS = [
        (r'\bos\.system\b', 'os.system 调用'),
        (r'\bsubprocess\.(call|run|Popen|check_call|check_output)\b', 'subprocess 调用'),
        (r'\beval\s*\(', 'eval 函数调用'),
        (r'\bexec\s*\(', 'exec 函数调用'),
        (r'\bcompile\s*\(', 'compile 函数调用'),
        (r'\b__import__\s*\(', '__import__ 调用'),
        (r'\bimport\s+sys\b', 'sys 模块导入'),
        (r'\bimport\s+os\b', 'os 模块导入'),
        (r'\bimport\s+subprocess\b', 'subprocess 模块导入'),
        (r'\bopen\s*\(', '文件 open 操作'),
        (r'\bread\s*\(', '文件 read 操作'),
        (r'\bwrite\s*\(', '文件 write 操作'),
        (r'\bsocket\.', 'socket 操作'),
        (r'\brequests\.', 'requests 库调用'),
        (r'\burlopen\b', 'URL 打开操作'),
        (r'\bchmod\b', 'chmod 操作'),
        (r'\bchown\b', 'chown 操作'),
        (r'\bmkdir\b', 'mkdir 操作'),
        (r'\brmdir\b', 'rmdir 操作'),
        (r'\bremove\b', 'remove 操作'),
        (r'\bunlink\b', 'unlink 操作'),
        (r'\bfork\b', 'fork 操作'),
        (r'\bexecv\b', 'execv 操作'),
        (r'\bpopen\b', 'popen 操作'),
    ]
    
    DANGEROUS_MODULES = [
        'sys', 'os', 'subprocess', 'shutil', 'glob', ' tempfile',
        'socket', 'urllib', 'requests', 'http', 'ftplib', 'telnetlib',
        'pty', 'resource', 'signal', 'multiprocessing'
    ]
    
    @classmethod
    def check_security(cls, code: str, language: str = 'python') -> Tuple[bool, List[str]]:
        warnings = []
        is_safe = True
        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(f"检测到潜在危险模式: {description}")
                is_safe = False
        if language == 'python':
            for module in cls.DANGEROUS_MODULES:
                import_pattern = rf'\bimport\s+{module}\b'
                from_pattern = rf'\bfrom\s+{module}\b'
                if re.search(import_pattern, code, re.IGNORECASE):
                    warnings.append(f"检测到危险模块导入: {module}")
                    is_safe = False
                if re.search(from_pattern, code, re.IGNORECASE):
                    warnings.append(f"检测到危险模块导入: {module}")
                    is_safe = False
        return is_safe, warnings


class CodeExecutor:
    LANGUAGE_COMMANDS = {
        'python': ['bash', '-c', 'python3 - <<\'EOF\'\n{}\nEOF'],
        'python3': ['bash', '-c', 'python3 - <<\'EOF\'\n{}\nEOF'],
        'javascript': ['bash', '-c', 'node -e \'{}\''],
        'java': ['bash', '-c', 'cat > Main.java <<\'EOF\'\n{}\nEOF\njavac Main.java && java Main'],
        'cpp': ['bash', '-c', 'cat > main.cpp <<\'EOF\'\n{}\nEOF\ng++ -std=c++17 main.cpp -o main && ./main'],
        'c': ['bash', '-c', 'cat > main.c <<\'EOF\'\n{}\nEOF\ngcc main.c -o main && ./main'],
    }
    
    def __init__(
        self,
        timeout: int = 10,
        max_memory_mb: int = 256,
        max_output_size: int = 1024 * 1024,
        working_dir: Optional[str] = None
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.working_dir = working_dir or tempfile.mkdtemp(prefix='code_exec_')
        self._execution_lock = Lock()
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
    
    def execute(self, code: str, language: str = 'python', input_data: str = '') -> Dict[str, Any]:
        start_time = time.time()
        output = ''
        error = ''
        return_code = -1
        memory_usage = 0
        try:
            if language not in self.LANGUAGE_COMMANDS:
                return {
                    'success': False, 'output': '', 'error': f'不支持的语言: {language}',
                    'execution_time': 0, 'memory_usage': 0, 'return_code': -1,
                    'timeout': False, 'oom': False
                }
            cmd_parts = self.LANGUAGE_COMMANDS[language]
            code_with_input = self._wrap_input(code, input_data, language)
            if len(cmd_parts) == 3:
                cmd_template, cmd_flag, cmd_args = cmd_parts
                if '{}' in cmd_args:
                    cmd = f"{cmd_template} {cmd_flag} {cmd_args.format(code_with_input)}"
                else:
                    cmd = f"{cmd_template} {cmd_flag} {cmd_args}"
            else:
                cmd_template, cmd_args = cmd_parts
                if '{}' in cmd_args:
                    cmd = [cmd_template, cmd_args.format(code_with_input)]
                else:
                    cmd = cmd_args.format(code_with_input)
            result = self._run_in_sandbox(cmd)
            output = result.get('stdout', '')[:self.max_output_size]
            error = result.get('stderr', '')[:self.max_output_size]
            return_code = result.get('returncode', -1)
            execution_time = result.get('execution_time', 0)
            timeout = result.get('timeout', False)
            oom = result.get('oom', False)
            if timeout:
                error = f'执行超时 (限制: {self.timeout}秒)\n{error}'
            if oom:
                error = f'内存超出限制 (限制: {self.max_memory_mb}MB)\n{error}'
            return {
                'success': return_code == 0 and not timeout and not oom,
                'output': output, 'error': error,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'return_code': return_code,
                'timeout': timeout, 'oom': oom
            }
        except Exception as e:
            return {
                'success': False, 'output': output,
                'error': f'执行异常: {str(e)}',
                'execution_time': time.time() - start_time,
                'memory_usage': memory_usage,
                'return_code': -1, 'timeout': False, 'oom': False
            }
    
    def _wrap_input(self, code: str, input_data: str, language: str = 'python', input_mode: str = 'code') -> str:
        if not input_data:
            return code
        if 'python' not in language.lower():
            return input_data
        if input_mode == 'code':
            return f"{code}\n\n{input_data}"
        else:
            wrapped_code = f"""
import sys
import io
input_data = '''{input_data}'''
sys.stdin = io.StringIO(input_data)
{code}
"""
            return wrapped_code
    
    def _run_in_sandbox(self, cmd: Union[str, List[str]]) -> Dict[str, Any]:
        with self._execution_lock:
            try:
                if isinstance(cmd, str):
                    shell = True
                    cmd_list = cmd
                else:
                    shell = False
                    cmd_list = cmd
                start_time = time.time()
                result = subprocess.run(
                    cmd_list,
                    shell=shell,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 5,
                    cwd=self.working_dir,
                    env={
                        **os.environ,
                        'PYTHONUNBUFFERED': '1',
                        'NODE_PATH': '',
                    }
                )
                execution_time = time.time() - start_time
                return {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'execution_time': execution_time,
                    'timeout': False,
                    'oom': False
                }
            except subprocess.TimeoutExpired:
                return {
                    'stdout': '', 'stderr': '执行超时',
                    'returncode': -1,
                    'execution_time': self.timeout,
                    'timeout': True,
                    'oom': False
                }
            except Exception as e:
                return {
                    'stdout': '', 'stderr': str(e),
                    'returncode': -1,
                    'execution_time': 0,
                    'timeout': False,
                    'oom': False
                }
    
    def cleanup(self):
        import shutil
        try:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
        except Exception:
            pass
    
    def __del__(self):
        self.cleanup()


class CodeEvaluator(BaseEvaluator):
    def __init__(self, config=None):
        super().__init__(config)
        self.executor = CodeExecutor(
            timeout=int(self.config.get("execution_timeout", 30)),
            max_memory_mb=int(self.config.get("max_memory_mb", 256)),
            working_dir=str(self.config.get("sandbox_working_dir", "/tmp/code_eval")),
        )
        self.supported_languages = ['python', 'python3', 'javascript', 'java', 'cpp', 'c']
    
    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        code = sample.get('code', '')
        language = sample.get('language', 'python')
        test_cases = sample.get('test_cases', [])
        
        if not code:
            return {
                'success': False, 'error': '代码为空',
                'test_results': [], 'pass_rate': 0.0
            }
        if language not in self.supported_languages:
            return {
                'success': False,
                'error': f'不支持的语言: {language}',
                'test_results': [], 'pass_rate': 0.0
            }
        if self.config.get("enable_security_check", True):
            is_safe, warnings = CodeSecurityChecker.check_security(code, language)
            if not is_safe:
                return {
                    'success': False,
                    'error': f'安全检查失败: {"; ".join(warnings)}',
                    'test_results': [], 'pass_rate': 0.0,
                    'security_warnings': warnings
                }
        test_results = []
        passed_count = 0
        for i, test_case in enumerate(test_cases):
            input_data = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            result = self.executor.execute(code, language, input_data)
            actual_output = result['output'].strip()
            expected_output_clean = expected_output.strip()
            is_pass = actual_output == expected_output_clean
            if is_pass:
                passed_count += 1
            test_results.append({
                'test_id': i + 1,
                'input': input_data,
                'expected_output': expected_output_clean,
                'actual_output': actual_output,
                'pass': is_pass,
                'execution_time': result['execution_time'],
                'error': result['error'] if not is_pass else ''
            })
        pass_rate = passed_count / len(test_results) if test_results else 0.0
        return {
            'success': pass_rate == 1.0,
            'pass_rate': pass_rate,
            'passed_count': passed_count,
            'total_count': len(test_results),
            'test_results': test_results,
            'execution_time': sum(t['execution_time'] for t in test_results),
            'security_check_passed': True
        }
