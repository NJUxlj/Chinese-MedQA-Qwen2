import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import PyPDF2
import pdfplumber
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PdfParser:
    """PDF解析器，用于解析PDF文件并提取文本和结构化信息"""

    def __init__(self):
        """初始化解析器"""
        self.supported_formats = ['.pdf']
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    def parse_to_documents(self, pdf_path: str) -> List[Document]:
        """将PDF解析为LangChain Document对象列表"""
        try:
            documents = []
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
            if pdf_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"不支持的文件格式: {pdf_path.suffix}")
            
            # 检查文件大小
            if pdf_path.stat().st_size > self.max_file_size:
                raise ValueError(f"PDF文件过大: {pdf_path.stat().st_size / 1024 / 1024:.2f}MB")
            
            # 提取文本内容
            text_content = self._extract_text_with_metadata(pdf_path)
            
            if not text_content['text'].strip():
                logger.warning(f"PDF文件 {pdf_path.name} 没有可提取的文本内容")
                return []
            
            # 创建Document对象
            doc = Document(
                page_content=text_content['text'],
                metadata={
                    'source': str(pdf_path),
                    'filename': pdf_path.name,
                    'file_size': pdf_path.stat().st_size,
                    'num_pages': text_content['num_pages'],
                    'extraction_method': text_content['method']
                }
            )
            
            documents.append(doc)
            logger.info(f"成功解析PDF文件: {pdf_path.name}, 文本长度: {len(text_content['text'])}")
            
            return documents
            
        except Exception as e:
            logger.error(f"解析PDF文件失败 {pdf_path}: {e}")
            return []

    def parse_to_text(self, pdf_path: str) -> str:
        """将PDF解析为纯文本"""
        try:
            documents = self.parse_to_documents(pdf_path)
            if documents:
                return documents[0].page_content
            return ""
        except Exception as e:
            logger.error(f"提取PDF文本失败 {pdf_path}: {e}")
            return ""

    def parse_to_kg_triplets(self, pdf_path: str) -> List[Dict[str, Any]]:
        """解析PDF文件，返回知识图谱三元组
        
        这个方法现在是一个占位符，实际的三元组提取逻辑在KGBuilder中实现
        """
        try:
            logger.info(f"开始从PDF提取知识图谱三元组: {pdf_path}")
            
            # 首先提取文本
            text = self.parse_to_text(pdf_path)
            
            if not text.strip():
                logger.warning(f"PDF文件 {pdf_path} 没有可提取的文本")
                return []
            
            # 这里可以添加简单的基于规则的实体提取
            # 完整的三元组提取逻辑在KGBuilder中实现
            
            logger.info(f"成功从PDF提取三元组: {pdf_path}")
            return []  # 返回空列表，实际实现请使用KGBuilder
            
        except Exception as e:
            logger.error(f"从PDF提取三元组失败 {pdf_path}: {e}")
            return []

    def _extract_text_with_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """提取PDF文本和元数据"""
        text_content = {
            'text': '',
            'num_pages': 0,
            'method': 'unknown'
        }
        
        # 尝试使用pdfplumber（更好的文本提取）
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content['num_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # 清理页面文本
                            cleaned_text = self._clean_page_text(page_text)
                            text_content['text'] += f"\n--- 第{page_num}页 ---\n"
                            text_content['text'] += cleaned_text
                    except Exception as e:
                        logger.warning(f"提取第{page_num}页文本失败: {e}")
                        continue
                
                if text_content['text'].strip():
                    text_content['method'] = 'pdfplumber'
                    logger.debug(f"使用pdfplumber成功提取文本，共{text_content['num_pages']}页")
                    return text_content
                    
        except Exception as e:
            logger.warning(f"pdfplumber提取失败: {e}")
        
        # 备用方案：使用PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content['num_pages'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # 清理页面文本
                            cleaned_text = self._clean_page_text(page_text)
                            text_content['text'] += f"\n--- 第{page_num}页 ---\n"
                            text_content['text'] += cleaned_text
                    except Exception as e:
                        logger.warning(f"提取第{page_num}页文本失败: {e}")
                        continue
                
                if text_content['text'].strip():
                    text_content['method'] = 'PyPDF2'
                    logger.debug(f"使用PyPDF2成功提取文本，共{text_content['num_pages']}页")
                    
        except Exception as e:
            logger.error(f"PyPDF2提取也失败: {e}")
        
        return text_content

    def _clean_page_text(self, text: str) -> str:
        """清理单页文本"""
        if not text:
            return ""
        
        # 移除多余的空行
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过空行和只有页码的行
            if line and not self._is_page_number_line(line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _is_page_number_line(self, line: str) -> bool:
        """判断是否为页码行"""
        # 简单的页码判断逻辑
        stripped = line.strip()
        
        # 纯数字行
        if stripped.isdigit():
            return True
        
        # 常见页码格式
        page_patterns = [
            r'^\d+$',                    # 纯数字
            r'^第\s*\d+\s*页$',          # 第X页
            r'^Page\s+\d+$',            # Page X
            r'^\d+\s*/\s*\d+$',         # X/Y
        ]
        
        import re
        for pattern in page_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        
        return False

    def batch_parse_pdfs(self, pdf_dir: str) -> List[Document]:
        """批量解析PDF目录中的所有PDF文件"""
        pdf_dir_path = Path(pdf_dir)
        
        if not pdf_dir_path.exists():
            raise FileNotFoundError(f"PDF目录不存在: {pdf_dir}")
        
        # 查找所有PDF文件
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始批量解析...")
        
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.parse_to_documents(str(pdf_file))
            all_documents.extend(documents)
        
        logger.info(f"批量解析完成，共解析 {len(all_documents)} 个文档")
        return all_documents

    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """获取PDF文件的基本信息"""
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
            info = {
                'file_path': str(pdf_path),
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'file_size_mb': round(pdf_path.stat().st_size / 1024 / 1024, 2),
                'extension': pdf_path.suffix.lower()
            }
            
            # 尝试获取页数
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    info['num_pages'] = len(pdf.pages)
            except:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        info['num_pages'] = len(pdf_reader.pages)
                except:
                    info['num_pages'] = 0
            
            return info
            
        except Exception as e:
            logger.error(f"获取PDF信息失败 {pdf_path}: {e}")
            return {}
