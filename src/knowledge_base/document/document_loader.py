# knowledge_base/document_loader.py  
import os  
from typing import List, Dict, Union, Optional  
import logging  
from pathlib import Path  

# Common document formats support  
from langchain_community.document_loaders import (  
    PyPDFLoader,  
    TextLoader,  
    UnstructuredWordDocumentLoader,  
    UnstructuredMarkdownLoader,  
    CSVLoader,  
    UnstructuredExcelLoader,  
    UnstructuredHTMLLoader  
)  

# For processing web content  
import requests  
from bs4 import BeautifulSoup  
from langchain.schema import Document  

logger = logging.getLogger(__name__)  

class DocumentLoader:  
    """  
    A document loader that supports multiple file formats for the medical knowledge base.  
    Supports: PDF, TXT, DOCX, MD, CSV, XLSX, HTML, and web URLs.  
    """  
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):  
        """  
        Initialize the document loader.  
        
        Args:  
            chunk_size: The size of text chunks for splitting documents  
            chunk_overlap: The overlap between consecutive chunks  
        """  
        self.chunk_size = chunk_size  
        self.chunk_overlap = chunk_overlap  
        
        # Map file extensions to appropriate loaders  
        self.loader_map = {  
            ".pdf": PyPDFLoader,  
            ".txt": TextLoader,  
            ".docx": UnstructuredWordDocumentLoader,  
            ".doc": UnstructuredWordDocumentLoader,  
            ".md": UnstructuredMarkdownLoader,  
            ".csv": CSVLoader,  
            ".xlsx": UnstructuredExcelLoader,  
            ".xls": UnstructuredExcelLoader,  
            ".html": UnstructuredHTMLLoader,  
            ".htm": UnstructuredHTMLLoader,  
        }  
    
    def load_document(self, file_path: str) -> List[Document]:  
        """  
        Load a single document from a file path.  
        
        Args:  
            file_path: Path to the document file  
            
        Returns:  
            List of Document objects  
        """  
        file_ext = os.path.splitext(file_path)[1].lower()  
        
        if file_ext not in self.loader_map:  
            raise ValueError(f"Unsupported file type: {file_ext}")  
        
        loader_class = self.loader_map[file_ext]  
        loader = loader_class(file_path)  
        
        try:  
            documents = loader.load()  
            logger.info(f"Successfully loaded document: {file_path}")  
            return documents  
        except Exception as e:  
            logger.error(f"Failed to load document {file_path}: {str(e)}")  
            raise  
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:  
        """  
        Load multiple documents from a list of file paths.  
        
        Args:  
            file_paths: List of paths to document files  
            
        Returns:  
            List of Document objects  
        """  
        all_documents = []  
        for file_path in file_paths:  
            try:  
                documents = self.load_document(file_path)  
                all_documents.extend(documents)  
            except Exception as e:  
                logger.warning(f"Skipping {file_path} due to error: {str(e)}")  
                continue  
                
        return all_documents  
    
    def load_from_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:  
        """  
        Load all supported documents from a directory.  
        
        Args:  
            directory_path: Path to the directory containing documents  
            recursive: Whether to recursively search subdirectories  
            
        Returns:  
            List of Document objects  
        """  
        all_documents = []  
        supported_extensions = set(self.loader_map.keys())  
        
        for root, _, files in os.walk(directory_path):  
            if not recursive and root != directory_path:  
                continue  
                
            for file in files:  
                file_ext = os.path.splitext(file)[1].lower()  
                if file_ext in supported_extensions:  
                    file_path = os.path.join(root, file)  
                    try:  
                        documents = self.load_document(file_path)  
                        all_documents.extend(documents)  
                    except Exception as e:  
                        logger.warning(f"Skipping {file_path} due to error: {str(e)}")  
                        continue  
        
        return all_documents  
    
    def load_from_web(self, url: str) -> List[Document]:  
        """  
        Load content from a web URL.  
        
        Args:  
            url: Web URL to fetch content from  
            
        Returns:  
            List of Document objects  
        """  
        try:  
            response = requests.get(url, timeout=10)  
            response.raise_for_status()  
            
            # Extract text content  
            soup = BeautifulSoup(response.content, 'html.parser')  
            
            # Remove script and style elements  
            for script in soup(["script", "style"]):  
                script.extract()  
                
            # Get text content  
            text = soup.get_text(separator="\n")  
            
            # Create a Document  
            metadata = {"source": url, "title": soup.title.string if soup.title else url}  
            document = Document(page_content=text, metadata=metadata)  
            
            return [document]  
            
        except Exception as e:  
            logger.error(f"Failed to load content from URL {url}: {str(e)}")  
            raise  
        
    def _is_valid_url(self, url: str) -> bool:  
        """Check if a string is a valid URL."""  
        try:  
            result = requests.head(url, timeout=5)  
            return result.status_code == 200  
        except:  
            return False  
            
    def load_from_sources(self, sources: List[str]) -> List[Document]:  
        """  
        Load documents from a list of sources that can be file paths, URLs, or directories.  
        
        Args:  
            sources: List of sources (file paths, URLs, or directories)  
            
        Returns:  
            List of Document objects  
        """  
        all_documents = []  
        
        for source in sources:  
            try:  
                if os.path.isfile(source):  
                    documents = self.load_document(source)  
                    all_documents.extend(documents)  
                elif os.path.isdir(source):  
                    documents = self.load_from_directory(source)  
                    all_documents.extend(documents)  
                elif self._is_valid_url(source):  
                    documents = self.load_from_web(source)  
                    all_documents.extend(documents)  
                else:  
                    logger.warning(f"Unsupported source type: {source}")  
            except Exception as e:  
                logger.error(f"Error loading from source {source}: {str(e)}")  
                continue  
                
        return all_documents  