
import os
import glob
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class LocalKnowledgeBase:
    """本地知识库处理器，用于处理和索引本地医疗知识文档"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地知识库。
        
        Args:
            config: 本地知识库配置字典
        """
        self.config = config
        self.directories = config.get('directories', [])
        self.file_types = config.get('file_types', ['.pdf', '.txt', '.csv', '.xlsx'])
        
        # 文本分割器，用于将长文档分割成适合向量化的块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def load_documents(self) -> List[Document]:
        """
        加载指定目录中的所有文档。
        
        Returns:
            文档对象列表
        """
        all_documents = []
        
        for directory in self.directories:
            if not os.path.exists(directory):
                print(f"警告: 目录不存在 {directory}")
                continue
                
            # 遍历所有支持的文件类型
            for file_type in self.file_types:
                file_pattern = os.path.join(directory, f"*{file_type}")
                files = glob.glob(file_pattern)
                
                print(f"在 {directory} 中找到 {len(files)} 个 {file_type} 文件")
                
                for file_path in tqdm(files, desc=f"处理{file_type}文件"):
                    try:
                        documents = self._load_file(file_path)
                        if documents:
                            all_documents.extend(documents)
                    except Exception as e:
                        print(f"加载文件 {file_path} 时出错: {e}")
                        
        print(f"总共加载了 {len(all_documents)} 个文档")
        return all_documents
        
    def _load_file(self, file_path: str) -> List[Document]:
        """
        根据文件类型加载单个文件。
        
        Args:
            file_path: 文件路径
            
        Returns:
            从文件加载的文档对象列表
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
                documents = loader.load()
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
                documents = loader.load()
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
            else:
                print(f"不支持的文件类型: {file_extension}")
                return []
                
            # 添加文件元数据
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['file_type'] = file_extension
                doc.metadata['filename'] = os.path.basename(file_path)
                
            return documents
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []
            
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成较小的块以便于向量化和检索。
        
        Args:
            documents: 要分割的文档列表
            
        Returns:
            分割后的文档块列表
        """
        return self.text_splitter.split_documents(documents)
        
    def process_documents(self) -> List[Document]:
        """
        完整的文档处理流程：加载并分割文档。
        
        Returns:
            处理后的文档块列表，准备进行向量化
        """
        documents = self.load_documents()
        split_docs = self.split_documents(documents)
        return split_docs
