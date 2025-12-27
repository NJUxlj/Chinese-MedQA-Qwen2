
import os
import time
from typing import Dict, List, Optional, Union, Any

from Bio import Entrez
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

class PubMedRetriever:
    """PubMed检索器，用于从PubMed获取医学文献信息"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PubMed检索器。
        
        Args:
            config: PubMed配置字典
        """
        self.config = config
        self.email = config.get('email')
        self.api_key = config.get('api_key', os.environ.get('PUBMED_API_KEY'))
        self.tool_name = config.get('tool_name', 'MedicalAIAgent')
        self.max_results = config.get('max_results', 10)
        
        # 设置Entrez
        Entrez.email = self.email
        Entrez.tool = self.tool_name
        if self.api_key:
            Entrez.api_key = self.api_key
            
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        在PubMed中搜索文献。
        
        Args:
            query: 搜索查询字符串
            max_results: 最大返回结果数（可选），默认使用配置值
            
        Returns:
            包含搜索结果的字典列表
        """
        max_results = max_results or self.max_results
        
        # 执行搜索以获取ID列表
        print(f"正在PubMed中搜索: {query}")
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results["IdList"]
            
            if not id_list:
                print("未找到匹配的PubMed文章")
                return []
                
            # 获取文章详情
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="xml",
                retmode="text"
            )
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            results = []
            # 解析文章数据
            for i, article in enumerate(articles['PubmedArticle']):
                try:
                    article_data = self._parse_article(article)
                    results.append(article_data)
                except Exception as e:
                    print(f"解析文章 {i+1} 时出错: {e}")
                    
            return results
            
        except Exception as e:
            print(f"PubMed搜索出错: {e}")
            return []
            
    def fetch_article_by_id(self, pmid: str) -> Optional[Dict]:
        """
        通过PubMed ID获取单篇文章。
        
        Args:
            pmid: PubMed ID
            
        Returns:
            包含文章信息的字典，如果未找到则返回None
        """
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="xml",
                retmode="text"
            )
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            if not articles.get('PubmedArticle'):
                return None
                
            article = articles['PubmedArticle'][0]
            return self._parse_article(article)
            
        except Exception as e:
            print(f"获取PubMed文章 {pmid} 时出错: {e}")
            return None
            
    def fetch_full_text(self, pmid: str) -> Optional[str]:
        """
        尝试获取文章的全文（如果可用）。
        
        Args:
            pmid: PubMed ID
            
        Returns:
            文章全文字符串，如果不可用则返回None
        """
        try:
            # 尝试从PMC获取全文链接
            pmc_link = self._get_pmc_link(pmid)
            
            if pmc_link:
                # 获取PMC全文
                response = requests.get(pmc_link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    article_text = soup.find('div', {'class': 'article-body'})
                    
                    if article_text:
                        # 提取正文
                        full_text = ""
                        sections = article_text.find_all(['h2', 'h3', 'p'])
                        for section in sections:
                            if section.name in ['h2', 'h3']:
                                full_text += f"\n\n{section.get_text()}\n"
                            else:
                                full_text += f"{section.get_text()}\n"
                        
                        return full_text
            
            return None
            
        except Exception as e:
            print(f"获取全文时出错: {e}")
            return None
            
    def _get_pmc_link(self, pmid: str) -> Optional[str]:
        """获取PMC全文链接"""
        try:
            search_handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=pmid
            )
            results = Entrez.read(search_handle)
            search_handle.close()
            
            # 检查是否有PMC链接
            if results[0]["LinkSetDb"]:
                pmc_id = results[0]["LinkSetDb"][0]["Link"][0]["Id"]
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}"
                
            return None
            
        except Exception as e:
            print(f"获取PMC链接时出错: {e}")
            return None
    
    def _parse_article(self, article) -> Dict:
        """
        解析PubMed文章XML数据。
        
        Args:
            article: PubMed文章XML对象
            
        Returns:
            包含文章数据的字典
        """
        article_data = {}
        
        # 提取PubMed ID
        try:
            article_data['pmid'] = article['MedlineCitation']['PMID']
        except (KeyError, IndexError):
            article_data['pmid'] = "未知"
            
        # 提取标题
        try:
            article_data['title'] = article['MedlineCitation']['Article']['ArticleTitle']
        except (KeyError, IndexError):
            article_data['title'] = "未知标题"
            
        # 提取摘要
        try:
            abstract_parts = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            if isinstance(abstract_parts, list):
                # 如果摘要有多个部分（例如有标签）
                abstract_text = ""
                for part in abstract_parts:
                    # 检查是否有标签
                    if hasattr(part, 'attributes') and 'Label' in part.attributes:
                        abstract_text += f"{part.attributes['Label']}: {part}\n"
                    else:
                        abstract_text += f"{part}\n"
                article_data['abstract'] = abstract_text
            else:
                article_data['abstract'] = abstract_parts
        except (KeyError, IndexError):
            article_data['abstract'] = "摘要不可用"
            
        # 提取发表日期
        try:
            pub_date = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
            date_str = ""
            
            if 'Year' in pub_date:
                date_str += pub_date['Year']
            if 'Month' in pub_date:
                date_str += f"-{pub_date['Month']}"
            if 'Day' in pub_date:
                date_str += f"-{pub_date['Day']}"
                
            article_data['publication_date'] = date_str
        except (KeyError, IndexError):
            article_data['publication_date'] = "未知日期"
            
        # 提取作者
        try:
            author_list = article['MedlineCitation']['Article']['AuthorList']
            authors = []
            
            for author in author_list:
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['LastName']} {author['ForeName']}")
                elif 'LastName' in author:
                    authors.append(author['LastName'])
                elif 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])
                    
            article_data['authors'] = ", ".join(authors) if authors else "未知作者"
        except (KeyError, IndexError):
            article_data['authors'] = "未知作者"
            
        # 提取期刊信息
        try:
            article_data['journal'] = article['MedlineCitation']['Article']['Journal']['Title']
        except (KeyError, IndexError):
            article_data['journal'] = "未知期刊"
            
        # 提取DOI
        try:
            for id_obj in article['PubmedData']['ArticleIdList']:
                if id_obj.attributes.get('IdType') == 'doi':
                    article_data['doi'] = str(id_obj)
                    break
            if 'doi' not in article_data:
                article_data['doi'] = "未知DOI"
        except (KeyError, IndexError):
            article_data['doi'] = "未知DOI"
            
        # 生成PubMed链接
        article_data['pubmed_link'] = f"https://pubmed.ncbi.nlm.nih.gov/{article_data['pmid']}/"
        
        return article_data







def run():
    pass





if __name__ == "__main__":
    run()