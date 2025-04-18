# knowledge_base/document_processor.py  
import re  
import html  
import logging  
from typing import List, Dict, Any, Optional  
import unicodedata  
import jieba  
from langchain.schema import Document  
from langchain.text_splitter import RecursiveCharacterTextSplitter  

logger = logging.getLogger(__name__)  

# This class handles document chunking, cleaning, and preparation for embedding:
class DocumentProcessor:  
    """  
    Processes documents for the medical knowledge base by cleaning text,  
    splitting into chunks, and enhancing metadata.  
    """  
    
    def __init__(  
        self,  
        chunk_size: int = 500,  
        chunk_overlap: int = 50,  
        separators: List[str] = ["\n\n", "\n", " ", ""],  
        keep_separator: bool = True,  
        min_chunk_size: int = 50  
    ):  
        """  
        Initialize the document processor.  
        
        Args:  
            chunk_size: The size of text chunks for splitting  
            chunk_overlap: The overlap between consecutive chunks  
            separators: List of separators for text splitting  
            keep_separator: Whether to keep the separator in chunks  
            min_chunk_size: Minimum size for a valid chunk  
        """  
        self.chunk_size = chunk_size  
        self.chunk_overlap = chunk_overlap  
        self.separators = separators  
        self.keep_separator = keep_separator  
        self.min_chunk_size = min_chunk_size  
        
        self.text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=self.chunk_size,  
            chunk_overlap=self.chunk_overlap,  
            separators=self.separators,  
            keep_separator=self.keep_separator  
        )  
    
    def clean_text(self, text: str) -> str:  
        """  
        Clean and normalize text content.  
        
        Args:  
            text: The text to clean  
            
        Returns:  
            Cleaned text  
        """  
        if not text:  
            return ""  
            
        # Decode HTML entities  
        text = html.unescape(text)  
        
        # Normalize unicode characters  
        text = unicodedata.normalize('NFKC', text)  
        
        # Remove excessive whitespace  
        text = re.sub(r'\s+', ' ', text)  
        
        # Remove control characters except newlines and tabs  
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  
        
        return text.strip()  
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:  
        """  
        Extract keywords from text using jieba for Chinese text.  
        
        Args:  
            text: The text to extract keywords from  
            top_k: Number of top keywords to extract  
            
        Returns:  
            List of keywords  
        """  
        # Use jieba to extract keywords for Chinese text  
        words = jieba.analyse.extract_tags(text, topK=top_k)  
        return words  
    
    def split_document(self, document: Document) -> List[Document]:  
        """  
        Split a document into chunks.  
        
        Args:  
            document: The document to split  
            
        Returns:  
            List of document chunks  
        """  
        # Clean the document text  
        cleaned_text = self.clean_text(document.page_content)  
        document.page_content = cleaned_text  
        
        # Split the document  
        chunks = self.text_splitter.split_documents([document])  
        
        # Filter out chunks that are too small  
        valid_chunks = [chunk for chunk in chunks if len(chunk.page_content) >= self.min_chunk_size]  
        
        # Add additional metadata  
        for i, chunk in enumerate(valid_chunks):  
            # Update metadata with chunk information  
            chunk.metadata.update({  
                "chunk_id": i,  
                "chunk_size": len(chunk.page_content),  
                "total_chunks": len(valid_chunks),  
                "keywords": self.extract_keywords(chunk.page_content)  
            })  
        
        return valid_chunks  
    
    def process_documents(self, documents: List[Document]) -> List[Document]:  
        """  
        Process a list of documents by cleaning and splitting them.  
        
        Args:  
            documents: List of documents to process  
            
        Returns:  
            List of processed document chunks  
        """  
        all_chunks = []  
        
        for i, doc in enumerate(documents):  
            try:  
                # Update document with index for tracking  
                doc.metadata["doc_id"] = i  
                doc.metadata["doc_total"] = len(documents)  
                
                # Split the document  
                chunks = self.split_document(doc)  
                all_chunks.extend(chunks)  
                
                logger.info(f"Document {i+1}/{len(documents)} split into {len(chunks)} chunks")  
            except Exception as e:  
                logger.error(f"Error processing document {i+1}: {str(e)}")  
                continue  
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")  
        return all_chunks  
    
    def enhance_document_metadata(self, document: Document, additional_metadata: Dict[str, Any]) -> Document:  
        """  
        Enhance a document with additional metadata.  
        
        Args:  
            document: The document to enhance  
            additional_metadata: Additional metadata to add  
            
        Returns:  
            Enhanced document  
        """  
        document.metadata.update(additional_metadata)  
        return document  
    
    def preprocess_chinese_text(self, text: str) -> str:  
        """  
        Specialized preprocessing for Chinese medical text.  
        
        Args:  
            text: Chinese text to preprocess  
            
        Returns:  
            Preprocessed text  
        """  
        # Remove spaces between Chinese characters (often incorrectly added during OCR)  
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)  
        
        # Normalize punctuation  
        text = text.replace('．', '.')  
        text = text.replace('，', ',')  
        text = text.replace('：', ':')  
        text = text.replace('；', ';')  
        text = text.replace('？', '?')  
        text = text.replace('！', '!')  
        
        return text  
    
    def segment_medical_content(self, text: str) -> List[str]:  
        """  
        Segment medical content based on common section headers.  
        
        Args:  
            text: Medical text to segment  
            
        Returns:  
            List of text segments  
        """  
        # Common medical document section patterns  
        section_patterns = [  
            r'病史[:：]',  
            r'主诉[:：]',  
            r'现病史[:：]',  
            r'既往史[:：]',  
            r'个人史[:：]',  
            r'家族史[:：]',  
            r'检查结果[:：]',  
            r'诊断[:：]',  
            r'治疗方案[:：]',  
            r'用药建议[:：]',  
            r'手术记录[:：]',  
            r'随访记录[:：]'  
        ]  
        
        # Combine patterns  
        pattern = '|'.join(section_patterns)  
        
        # Split by section headers  
        sections = re.split(f'({pattern})', text)  
        
        # Recombine headers with their content  
        result = []  
        for i in range(0, len(sections)-1, 2):  
            if i+1 < len(sections):  
                result.append(sections[i] + sections[i+1])  
            else:  
                result.append(sections[i])  
                
        # Handle case where text doesn't start with a header  
        if not re.match(pattern, sections[0]) and sections[0].strip():  
            result.insert(0, sections[0])  
            
        return result  