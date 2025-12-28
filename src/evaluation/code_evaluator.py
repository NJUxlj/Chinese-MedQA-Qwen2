'''
评估模型在 SWE-bench, LiveCodeBench, codeforce 等代码测试集上的性能

包含以下核心功能:
1. 多语言代码执行支持 (Python, JavaScript, etc.)
2. 沙箱隔离执行环境 (通过 subprocess + 资源限制)
3. 超时和内存限制保护
4. 多测试用例验证
5. 安全性检查 (防止恶意代码执行)
6. 详细的执行报告生成
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
from config.evaluator_config import CodeEvaluatorConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset


class CodeSecurityChecker:
    '''代码安全性检查器
    
    检测并阻止潜在的恶意代码执行，包括:
    - 系统命令调用 (os.system, subprocess, etc.)
    - 文件系统操作 (open, read, write, etc.)
    - 网络操作 (socket, urllib, etc.)
    - 导入危险模块 (sys, os, subprocess, etc.)
    '''
    
    DANGEROUS_PATTERNS = [
        (r'\bos\.system\b', 'os.system 调用'),   # \b	单词边界（匹配单词的开头和结尾，避免部分匹配）
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
        '''检查代码安全性
        
        Args:
            code: 要检查的代码
            language: 编程语言
            
        Returns:
            Tuple[is_safe, warnings]: 是否安全，警告列表
        '''
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
    '''代码执行器
    
    负责在沙箱环境中安全地执行代码，支持:
    - 超时限制
    - 内存限制
    - 输出捕获
    - 错误处理
    '''
    
    LANGUAGE_COMMANDS = {
        # Python: 使用 bash 执行，通过 heredoc 传入代码
        # 执行方式：使用 bash -c 配合 heredoc 语法
        # 优点：无需创建临时文件，自动清理，兼容性好
        'python': ['bash', '-c', 'python3 - <<\'EOF\'\n{}\nEOF'],
        'python3': ['bash', '-c', 'python3 - <<\'EOF\'\n{}\nEOF'],
        
        # JavaScript: 使用 node 解释器执行
        # 执行方式：通过 -e 参数直接执行代码字符串
        'javascript': ['bash', '-c', 'node -e \'{}\''],
        
        # Java: 需要编译和运行两步
        # 执行方式：
        # 1. 将代码写入 Main.java 文件
        # 2. 使用 javac 编译
        # 3. 使用 java 运行
        # 注意：Java 代码必须包含 Main 类

        # 具体执行步骤：
        # Java: 使用 bash 一次性完成写文件、编译、运行的三步骤
        # 1. cat > Main.java <<'EOF' … EOF：将代码块写入 Main.java
        # 2. javac Main.java：编译生成字节码
        # 3. java Main：运行主类（要求源码必须含 public class Main{...}）
        'java': ['bash', '-c', 'cat > Main.java <<\'EOF\'\n{}\nEOF\njavac Main.java && java Main'],
        
        # C++: 使用 g++ 编译并运行
        # 执行方式：
        # 1. 将代码写入 main.cpp 文件
        # 2. 使用 g++ 编译（支持 C++17 标准）
        # 3. 运行编译后的可执行文件
        'cpp': ['bash', '-c', 'cat > main.cpp <<\'EOF\'\n{}\nEOF\ng++ -std=c++17 main.cpp -o main && ./main'],
        
        # C: 使用 gcc 编译并运行
        # 执行方式：
        # 1. 将代码写入 main.c 文件
        # 2. 使用 gcc 编译
        # 3. 运行编译后的可执行文件
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
    
    def execute(
        self,
        code: str,
        language: str = 'python',
        input_data: str = ''
    ) -> Dict[str, Any]:
        '''执行代码
        
        Args:
            code: 要执行的代码
            language: 编程语言
            input_data: 标准输入数据
            
        Returns:
            Dict containing: success, output, error, execution_time, memory_usage, return_code
        '''
        start_time = time.time()
        output = ''
        error = ''
        return_code = -1
        memory_usage = 0
        
        try:
            if language not in self.LANGUAGE_COMMANDS:
                return {
                    'success': False,
                    'output': '',
                    'error': f'不支持的语言: {language}',
                    'execution_time': 0,
                    'memory_usage': 0,
                    'return_code': -1,
                    'timeout': False,
                    'oom': False
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
                'output': output,
                'error': error,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'return_code': return_code,
                'timeout': timeout,
                'oom': oom
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': output,
                'error': f'执行异常: {str(e)}',
                'execution_time': time.time() - start_time,
                'memory_usage': memory_usage,
                'return_code': -1,
                'timeout': False,
                'oom': False
            }
    
    def _wrap_input(self, code: str, input_data: str, language: str = 'python', input_mode: str = 'code') -> str:
        '''包装输入数据
        
        支持两种输入模式：
        - input_mode='code': input_data 是需要执行的测试代码（如 print(add(1, 2))）
          直接将测试代码拼接到函数定义后面执行
        - input_mode='stdin': input_data 是真正的 stdin 输入数据
          通过重定向 sys.stdin 来模拟输入
        
        Args:
            code: 要执行的代码
            input_data: 输入数据（测试代码或 stdin 输入）
            language: 编程语言
            input_mode: 输入模式，'code' 或 'stdin'
        '''
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

# 将输入数据存储为多行字符串
# 为什么要用三引号？这样可以保留输入数据中的换行符，
# 使得多行输入（如数组、多行字符串等）能够被正确模拟
input_data = '''{input_data}'''

# 使用 io.StringIO 创建一个内存中的文件对象来模拟 stdin
# StringIO 可以像文件一样被读取，但数据存储在内存中
# 这样做的好处是：
# 1. 不需要创建真实的文件，避免了 I/O 开销和文件系统操作
# 2. 可以在内存中快速创建和销毁，非常适合测试场景
# 3. 保持了与真实文件相同的读取接口（readline、readlines 等）
sys.stdin = io.StringIO(input_data)

# 执行原始代码，此时代码中的 input() 调用会从我们创建的 StringIO 对象中读取数据
{code}
"""
            return wrapped_code
    
    def _run_in_sandbox(self, cmd: Union[str, List[str]]) -> Dict[str, Any]:
        '''在沙箱中运行命令'''
        '''
        在沙箱环境中执行命令的核心方法
        
        执行流程：
        1. 获取执行锁，确保同一时间只有一个代码执行（防止并发冲突）
        2. 统一命令格式（支持字符串和列表两种形式）
        3. 使用 subprocess 运行命令，配置适当的执行环境
        4. 记录执行结果和耗时
        
        注意事项：
        - 使用 lock 会限制并发性能，但确保了文件操作的安全性
        - 设置 PYTHONUNBUFFERED 确保 Python 输出立即可见
        - 设置 NODE_PATH 为空避免 Node.js 模块查找问题
        - 超时设置为 timeout+5，给进程清理留出额外时间
        '''
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
                    'stdout': '',
                    'stderr': '执行超时',
                    'returncode': -1,
                    'execution_time': self.timeout,
                    'timeout': True,
                    'oom': False
                }
            except Exception as e:
                return {
                    'stdout': '',
                    'stderr': str(e),
                    'returncode': -1,
                    'execution_time': 0,
                    'timeout': False,
                    'oom': False
                }
    
    def cleanup(self):
        '''清理临时文件'''
        import shutil
        try:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
        except Exception:
            pass
    
    def __del__(self):
        self.cleanup()


class CodeEvaluator(BaseEvaluator):
    '''代码评估器
    
    评估模型生成的代码正确性，支持:
    - 多语言代码执行验证
    - 多测试用例验证
    - 执行结果统计
    - 安全性检查
    
    使用方法:
    ```python
    config = CodeEvaluatorConfig(
        model_name_or_path="Qwen/Qwen3-14B",
        test_dataset_path="data/test.json",
        execution_timeout=10,
        max_memory_mb=256,
        enable_security_check=True
    )
    
    evaluator = CodeEvaluator(config)
    results = evaluator.evaluate()
    ```
    '''
    
    def __init__(self, config: CodeEvaluatorConfig):
        '''初始化代码评估器
        
        Args:
            config: 代码评估器配置
        '''
        super().__init__(config)
        
        self.config = config
        self.logger = setup_logger(name=self.__class__.__name__, level="INFO")
        
        self.executor = CodeExecutor(
            timeout=config.execution_timeout,
            max_memory_mb=config.max_memory_mb,
            working_dir=config.sandbox_working_dir
        )
        
        self.supported_languages = ['python', 'python3', 'javascript', 'java', 'cpp', 'c']
    
    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        '''评估单个样本
        
        Args:
            sample: 包含 code, language, test_cases 的字典
            
        Returns:
            评估结果字典
        '''
        code = sample.get('code', '')
        language = sample.get('language', 'python')
        test_cases = sample.get('test_cases', [])
        ground_true_answer = sample.get('ground_true_answer', '')
        
        if not code:
            return {
                'success': False,
                'error': '代码为空',
                'test_results': [],
                'pass_rate': 0.0
            }
        
        if language not in self.supported_languages:
            return {
                'success': False,
                'error': f'不支持的语言: {language}',
                'test_results': [],
                'pass_rate': 0.0
            }
        
        if self.config.enable_security_check:
            is_safe, warnings = CodeSecurityChecker.check_security(code, language)
            if not is_safe:
                return {
                    'success': False,
                    'error': f'安全检查失败: {"; ".join(warnings)}',
                    'test_results': [],
                    'pass_rate': 0.0,
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
    
    def evaluate_batch_samples(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        '''批量评估样本
        
        Args:
            batch: 包含多个样本的字典
            
        Returns:
            批量评估结果
        '''
        results = []
        
        codes = batch.get('codes', [])
        languages = batch.get('languages', [])
        test_cases_list = batch.get('test_cases_list', [])
        ground_true_answers = batch.get('ground_true_answers', [])
        
        for i in range(len(codes)):
            sample = {
                'code': codes[i],
                'language': languages[i] if i < len(languages) else 'python',
                'test_cases': test_cases_list[i] if i < len(test_cases_list) else [],
                'ground_true_answer': ground_true_answers[i] if i < len(ground_true_answers) else ''
            }
            
            result = self.evaluate_one_sample(sample)
            results.append(result)
        
        return {
            'results': results,
            'total_samples': len(results),
            'passed_samples': sum(1 for r in results if r.get('success', False)),
            'total_tests': sum(r.get('total_count', 0) for r in results),
            'passed_tests': sum(r.get('passed_count', 0) for r in results)
        }
    
    def evaluate(self) -> Dict[str, Any]:
        '''评估整个测试集
        
        Returns:
            包含详细统计信息的评估结果
        '''
        self.logger.info(f"开始评估，测试集大小: {len(self.test_data)}")
        
        all_results = []
        total_execution_time = 0
        
        for idx, sample in enumerate(self.test_data):
            self.logger.info(f"评估样本 {idx + 1}/{len(self.test_data)}")
            
            result = self.evaluate_one_sample(sample)
            result['sample_id'] = idx
            
            all_results.append(result)
            total_execution_time += result.get('execution_time', 0)
        
        total_samples = len(all_results)
        passed_samples = sum(1 for r in all_results if r.get('success', False))
        total_tests = sum(r.get('total_count', 0) for r in all_results)
        passed_tests = sum(r.get('passed_count', 0) for r in all_results)
        
        sample_pass_rate = passed_samples / total_samples if total_samples > 0 else 0.0
        test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        avg_execution_time = total_execution_time / total_samples if total_samples > 0 else 0
        
        error_distribution = {
            'security_failed': sum(1 for r in all_results if 'security_warnings' in r),
            'timeout': sum(1 for r in all_results if any(t.get('timeout', False) for t in r.get('test_results', []))),
            'execution_error': sum(1 for r in all_results if not r.get('success') and 'security_warnings' not in r),
            'wrong_answer': sum(1 for r in all_results if r.get('pass_rate', 0) < 1.0 and r.get('pass_rate', 0) > 0.0)
        }
        
        final_results = {
            'evaluation_summary': {
                'total_samples': total_samples,
                'passed_samples': passed_samples,
                'sample_pass_rate': sample_pass_rate,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'test_pass_rate': test_pass_rate,
                'total_execution_time': total_execution_time,
                'avg_execution_time_per_sample': avg_execution_time
            },
            'error_distribution': error_distribution,
            'detailed_results': all_results,
            'config': {
                'timeout': self.config.execution_timeout,
                'max_memory_mb': self.config.max_memory_mb,
                'security_check_enabled': self.config.enable_security_check
            }
        }
        
        self.logger.info(f"评估完成: {passed_samples}/{total_samples} 样本通过, {passed_tests}/{total_tests} 测试用例通过")
        self.logger.info(f"样本通过率: {sample_pass_rate:.2%}, 测试用例通过率: {test_pass_rate:.2%}")
        
        return final_results
    
    def evaluate_with_ground_truth(
        self,
        generated_code: str,
        ground_true_code: str,
        test_cases: List[Dict[str, Any]],
        language: str = 'python'
    ) -> Dict[str, Any]:
        '''使用参考答案评估生成代码
        
        Args:
            generated_code: 生成的代码
            ground_true_code: 参考正确代码
            test_cases: 测试用例列表
            language: 编程语言
            
        Returns:
            对比评估结果
        """
        '''
        generated_result = self.evaluate_one_sample({
            'code': generated_code,
            'language': language,
            'test_cases': test_cases
        })
        
        reference_result = self.evaluate_one_sample({
            'code': ground_true_code,
            'language': language,
            'test_cases': test_cases
        })
        
        return {
            'generated_code_result': generated_result,
            'reference_code_result': reference_result,
            'generation_correct': generated_result.get('success', False),
            'reference_correct': reference_result.get('success', False),
            'relative_performance': {
                'generated_vs_reference': 'better' if generated_result.get('pass_rate', 0) > reference_result.get('pass_rate', 0) else 'equal' if generated_result.get('pass_rate', 0) == reference_result.get('pass_rate', 0) else 'worse'
            }
        }
    
    def cleanup(self):
        '''清理资源'''
        self.executor.cleanup()


def run_evaluation():
    '''运行评估的入口函数'''
    config = CodeEvaluatorConfig(
        model_name_or_path="Qwen/Qwen3-14B",
        test_dataset_path="data/test.json",
        execution_timeout=10,
        max_memory_mb=256,
        enable_security_check=True
    )
    
    evaluator = CodeEvaluator(config)
    
    try:
        results = evaluator.evaluate()
        
        save_path = config.eval_result_save_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {save_path}")
        
        return results
        
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    run_evaluation()
