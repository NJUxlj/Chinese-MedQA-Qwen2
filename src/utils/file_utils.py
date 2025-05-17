# utils/file_utils.py

import os
import json
import yaml
import pickle
import csv
import shutil
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO, TextIO
import time
from pathlib import Path
import uuid
import gzip
import zipfile
import tarfile
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FileManager:
    """文件管理器，处理文件操作"""
    
    @staticmethod
    def ensure_dir(directory: str) -> str:
        """确保目录存在，如果不存在则创建"""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        return directory
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """计算文件的哈希值"""
        hash_algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256
        }
        
        if algorithm not in hash_algorithms:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
        
        hash_obj = hash_algorithms[algorithm]()
        
        with open(file_path, "rb") as f:
            # 分块读取以处理大文件
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: str, human_readable: bool = False) -> Union[int, str]:
        """获取文件大小，可选返回人类可读格式"""
        size = os.path.getsize(file_path)
        
        if not human_readable:
            return size
        
        # 转换为人类可读格式
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024 or unit == "TB":
                return f"{size:.2f} {unit}"
            size /= 1024
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """获取文件扩展名（不带点）"""
        return os.path.splitext(file_path)[1].lstrip(".")
    
    @staticmethod
    def get_file_modification_time(file_path: str, as_timestamp: bool = False) -> Union[float, str]:
        """获取文件最后修改时间"""
        mtime = os.path.getmtime(file_path)
        
        if as_timestamp:
            return mtime
        
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    
    @staticmethod
    def create_temp_file(suffix: Optional[str] = None, prefix: Optional[str] = None, 
                         dir: Optional[str] = None, text: bool = False) -> Tuple[TextIO, str]:
        """创建临时文件"""
        return tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
    
    @staticmethod
    def create_temp_dir(suffix: Optional[str] = None, prefix: Optional[str] = None, 
                        dir: Optional[str] = None) -> str:
        """创建临时目录"""
        return tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    
    @staticmethod
    def safe_filename(filename: str, replacement: str = "_") -> str:
        """生成安全的文件名（删除非法字符）"""
        # 替换Windows和类Unix系统中的非法字符
        illegal_chars = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
        
        for char in illegal_chars:
            filename = filename.replace(char, replacement)
        
        return filename
    
    @staticmethod
    def generate_unique_filename(base_path: str, extension: str) -> str:
        """生成唯一的文件名"""
        dir_path = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, _ = os.path.splitext(filename)
        
        # 确保扩展名带有点
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        
        unique_name = f"{name}_{uuid.uuid4().hex[:8]}{extension}"
        return os.path.join(dir_path, unique_name)
    
    @staticmethod
    def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
        """复制文件"""
        if os.path.exists(dst) and not overwrite:
            logger.warning(f"目标文件已存在且未设置覆盖: {dst}")
            return False
        
        try:
            shutil.copy2(src, dst)
            logger.info(f"文件已复制: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"复制文件错误: {e}")
            return False
    
    @staticmethod
    def move_file(src: str, dst: str, overwrite: bool = False) -> bool:
        """移动文件"""
        if os.path.exists(dst) and not overwrite:
            logger.warning(f"目标文件已存在且未设置覆盖: {dst}")
            return False
        
        try:
            shutil.move(src, dst)
            logger.info(f"文件已移动: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"移动文件错误: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: str, secure: bool = False) -> bool:
        """删除文件，可选安全删除（覆盖内容）"""
        if not os.path.exists(file_path):
            logger.warning(f"要删除的文件不存在: {file_path}")
            return False
        
        try:
            if secure and os.path.isfile(file_path):
                # 安全删除：先用随机数据覆盖文件内容
                file_size = os.path.getsize(file_path)
                with open(file_path, "wb") as f:
                    # 写入随机数据
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            
            logger.info(f"已删除: {file_path}")
            return True
        except Exception as e:
            logger.error(f"删除错误: {e}")
            return False


class DataFileHandler:
    """数据文件处理器，处理各种格式的数据文件"""
    
    @staticmethod
    def load_json(file_path: str) -> Any:
        """加载JSON文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件错误: {e}")
            raise
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: Optional[int] = 2, 
                 ensure_ascii: bool = False, sort_keys: bool = False) -> bool:
        """保存数据为JSON文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
            
            logger.info(f"JSON数据已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存JSON文件错误: {e}")
            return False
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Any]:
        """加载JSONL文件"""
        data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        data.append(json.loads(line))
            return data
        except Exception as e:
            logger.error(f"加载JSONL文件错误: {e}")
            raise
    
    @staticmethod
    def save_jsonl(data: List[Any], file_path: str, ensure_ascii: bool = False) -> bool:
        """保存数据为JSONL文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
            
            logger.info(f"JSONL数据已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存JSONL文件错误: {e}")
            return False
    
    @staticmethod
    def load_yaml(file_path: str) -> Any:
        """加载YAML文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载YAML文件错误: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Any, file_path: str) -> bool:
        """保存数据为YAML文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"YAML数据已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存YAML文件错误: {e}")
            return False
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """加载Pickle文件"""
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载Pickle文件错误: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any, file_path: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
        """保存数据为Pickle文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=protocol)
            
            logger.info(f"Pickle数据已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存Pickle文件错误: {e}")
            return False
    
    @staticmethod
    def load_csv(file_path: str, has_header: bool = True, delimiter: str = ",") -> List[Dict[str, str]]:
        """加载CSV文件为字典列表（第一行为键）"""
        try:
            data = []
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                if has_header:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        data.append(dict(row))
                else:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        data.append(row)
            return data
        except Exception as e:
            logger.error(f"加载CSV文件错误: {e}")
            raise
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], file_path: str, delimiter: str = ",") -> bool:
        """保存字典列表为CSV文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            # 获取所有可能的字段名
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"CSV数据已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存CSV文件错误: {e}")
            return False
    
    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> str:
        """加载文本文件"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载文本文件错误: {e}")
            raise
    
    @staticmethod
    def save_text(text: str, file_path: str, encoding: str = "utf-8") -> bool:
        """保存文本到文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "w", encoding=encoding) as f:
                f.write(text)
            
            logger.info(f"文本已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存文本文件错误: {e}")
            return False
    
    @staticmethod
    def load_text_lines(file_path: str, encoding: str = "utf-8", 
                      strip_lines: bool = True, skip_empty: bool = True) -> List[str]:
        """加载文本文件为行列表"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                if strip_lines:
                    lines = [line.strip() for line in f]
                else:
                    lines = [line.rstrip("\n") for line in f]
                
                if skip_empty:
                    lines = [line for line in lines if line]
                
                return lines
        except Exception as e:
            logger.error(f"加载文本行错误: {e}")
            raise
    
    @staticmethod
    def save_text_lines(lines: List[str], file_path: str, encoding: str = "utf-8") -> bool:
        """保存行列表到文本文件"""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                FileManager.ensure_dir(directory)
            
            with open(file_path, "w", encoding=encoding) as f:
                for line in lines:
                    f.write(line + "\n")
            
            logger.info(f"文本行已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存文本行错误: {e}")
            return False


class CompressedFileHandler:
    """压缩文件处理器"""
    
    @staticmethod
    def compress_file(file_path: str, output_path: Optional[str] = None, 
                     method: str = "gzip") -> str:
        """压缩文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if output_path is None:
            if method == "gzip":
                output_path = f"{file_path}.gz"
            elif method == "zip":
                output_path = f"{file_path}.zip"
            else:
                raise ValueError(f"不支持的压缩方法: {method}")
        
        try:
            if method == "gzip":
                with open(file_path, "rb") as f_in:
                    with gzip.open(output_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif method == "zip":
                with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, os.path.basename(file_path))
            
            else:
                raise ValueError(f"不支持的压缩方法: {method}")
            
            logger.info(f"文件已压缩: {file_path} -> {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"压缩文件错误: {e}")
            raise
    
    @staticmethod
    def decompress_file(file_path: str, output_path: Optional[str] = None) -> str:
        """解压文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            if file_path.endswith(".gz"):
                if output_path is None:
                    output_path = file_path[:-3]  # 移除.gz后缀
                
                with gzip.open(file_path, "rb") as f_in:
                    with open(output_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif file_path.endswith(".zip"):
                if output_path is None:
                    output_path = os.path.splitext(file_path)[0]
                
                with zipfile.ZipFile(file_path, "r") as zipf:
                    zipf.extractall(path=output_path)
            
            elif file_path.endswith((".tar", ".tar.gz", ".tgz")):
                if output_path is None:
                    output_path = os.path.splitext(file_path)[0]
                
                with tarfile.open(file_path, "r:*") as tarf:
                    tarf.extractall(path=output_path)
            
            else:
                raise ValueError(f"无法识别的压缩文件格式: {file_path}")
            
            logger.info(f"文件已解压: {file_path} -> {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"解压文件错误: {e}")
            raise
    
    @staticmethod
    def compress_directory(directory: str, output_path: Optional[str] = None, 
                         method: str = "zip") -> str:
        """压缩目录"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        if output_path is None:
            parent_dir = os.path.dirname(directory)
            dir_name = os.path.basename(directory)
            
            if method == "zip":
                output_path = os.path.join(parent_dir, f"{dir_name}.zip")
            elif method == "tar":
                output_path = os.path.join(parent_dir, f"{dir_name}.tar")
            elif method == "gztar":
                output_path = os.path.join(parent_dir, f"{dir_name}.tar.gz")
            else:
                raise ValueError(f"不支持的压缩方法: {method}")
        
        try:
            root_dir = os.path.dirname(directory)
            base_dir = os.path.basename(directory)
            
            if method == "zip":
                shutil.make_archive(os.path.splitext(output_path)[0], "zip", 
                                  root_dir=root_dir, base_dir=base_dir)
            elif method == "tar":
                shutil.make_archive(os.path.splitext(output_path)[0], "tar", 
                                  root_dir=root_dir, base_dir=base_dir)
            elif method == "gztar":
                shutil.make_archive(os.path.splitext(output_path)[0], "gztar", 
                                  root_dir=root_dir, base_dir=base_dir)
            else:
                raise ValueError(f"不支持的压缩方法: {method}")
            
            logger.info(f"目录已压缩: {directory} -> {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"压缩目录错误: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 文件管理示例
    fm = FileManager()
    test_dir = "test_dir"
    fm.ensure_dir(test_dir)
    
    # 创建和操作文件
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Hello, World!")
    
    # 文件信息
    print(f"File size: {fm.get_file_size(test_file, human_readable=True)}")
    print(f"File modification time: {fm.get_file_modification_time(test_file)}")
    print(f"File extension: {fm.get_file_extension(test_file)}")
    
    # 数据文件操作示例
    dfh = DataFileHandler()
    
    # JSON操作
    json_data = {"name": "Test", "value": 123}
    json_file = os.path.join(test_dir, "test.json")
    dfh.save_json(json_data, json_file)
    loaded_json = dfh.load_json(json_file)
    print(f"Loaded JSON: {loaded_json}")
    
    # 清理测试文件
    fm.delete_file(test_dir)
