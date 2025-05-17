# 使用爬虫来爬取全网的消化内科的病历数据
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
消化内科医疗数据爬虫
用于抓取全网消化内科相关的医疗数据，包括疾病信息、症状、治疗方案、药物信息等
"""

import os
import re
import time
import json
import random
import hashlib
import argparse
import logging
import datetime
import urllib.parse
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.exceptions import InsecureRequestWarning

import requests
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm
import yaml

# 忽略SSL警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MedicalCrawler")

# 消化内科相关关键词
GASTRO_KEYWORDS = [
    "消化内科", "胃肠", "消化系统", "肝胆", "胰腺", "食管", "胃病", "胃炎", 
    "胃溃疡", "十二指肠溃疡", "结肠炎", "胰腺炎", "肝炎", "肝硬化", 
    "肠易激综合征", "克罗恩", "溃疡性结肠炎", "消化不良", "胃食管反流", 
    "胆囊炎", "胆结石", "肠梗阻", "腹泻", "便秘", "黄疸", "腹痛", 
    "肠息肉", "结肠癌", "胃癌", "食管癌", "肝癌", "胆管癌", "胰腺癌",
    "胃镜", "肠镜", "内镜", "超声内镜", "ERCP", "肝功能", "幽门螺杆菌", 
    "消化系统出血", "腹水", "Barrett食管", "肝纤维化"
]

class MedicalDataCrawler:
    """消化内科医疗数据爬虫类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化爬虫
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 网站列表
        self.sites = self.config["sites"]
        
        # 爬虫参数
        self.delay = self.config["crawler"]["delay"]
        self.timeout = self.config["crawler"]["timeout"]
        self.max_retry = self.config["crawler"]["max_retry"]
        self.max_workers = self.config["crawler"]["max_workers"]
        self.respect_robots = self.config["crawler"]["respect_robots"]
        
        # 数据存储设置
        self.data_dir = Path(self.config["storage"]["data_dir"])
        self.file_format = self.config["storage"]["file_format"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 禁止访问的网站列表(robots.txt)
        self.disallowed_sites = set()
        
        # 已爬取的URL
        self.crawled_urls = set()
        
        # 加载已爬取的URL(如果有)
        self.crawled_urls_file = self.data_dir / "crawled_urls.json"
        if self.crawled_urls_file.exists():
            try:
                with open(self.crawled_urls_file, 'r', encoding='utf-8') as f:
                    self.crawled_urls = set(json.load(f))
                logger.info(f"已加载 {len(self.crawled_urls)} 个已爬取的URL")
            except Exception as e:
                logger.error(f"加载已爬取URL失败: {e}")
        
        # 创建UA生成器
        self.ua = UserAgent(browsers=['chrome', 'edge', 'firefox'])
        
        # 加载代理IP列表(如果配置了)
        self.proxies = self._load_proxies()
        
        # 统计信息
        self.stats = {
            "total_urls": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "extracted_data": 0,
            "start_time": time.time()
        }
        
        logger.info(f"爬虫初始化完成，目标站点数: {len(self.sites)}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        default_config = {
            "sites": [
                {
                    "name": "丁香园",
                    "url": "https://www.dxy.cn/",
                    "allowed_domains": ["dxy.cn"],
                    "start_urls": [
                        "https://www.dxy.cn/bbs/topic/33349", # 消化内科
                        "https://www.dxy.cn/bbs/topic/33351"  # 肝病
                    ],
                    "article_pattern": "article[class*='topic-item']",
                    "title_pattern": "h3[class*='topic-title']",
                    "content_pattern": "div[class*='topic-content']",
                    "next_page_pattern": "a[class*='next']",
                    "max_pages": 50
                },
                {
                    "name": "好大夫在线",
                    "url": "https://www.haodf.com/",
                    "allowed_domains": ["haodf.com"],
                    "start_urls": [
                        "https://www.haodf.com/keshi/2-xiaohuaneike.html",
                        "https://www.haodf.com/citiao/list-2.html"
                    ],
                    "article_pattern": "div[class*='disease-info']",
                    "title_pattern": "h3[class*='disease-name']",
                    "content_pattern": "div[class*='content-box']",
                    "next_page_pattern": "a[class*='next-page']",
                    "max_pages": 50
                },
                {
                    "name": "医脉通",
                    "url": "https://www.medlive.cn/",
                    "allowed_domains": ["medlive.cn"],
                    "start_urls": [
                        "https://news.medlive.cn/all/info-progress/show-144061_97.html",
                        "https://gi.medlive.cn/"
                    ],
                    "article_pattern": "div[class*='article-item']",
                    "title_pattern": "h3[class*='article-title']",
                    "content_pattern": "div[class*='article-content']",
                    "next_page_pattern": "a[class*='pagination-next']",
                    "max_pages": 30
                },
                {
                    "name": "人民健康网",
                    "url": "http://health.people.com.cn/",
                    "allowed_domains": ["health.people.com.cn"],
                    "start_urls": [
                        "http://health.people.com.cn/GB/408656/index.html" # 消化内科
                    ],
                    "article_pattern": "div[class*='list_item']",
                    "title_pattern": "h3",
                    "content_pattern": "div[id*='content_area']",
                    "next_page_pattern": "a[class*='next']",
                    "max_pages": 30
                },
                {
                    "name": "39健康网",
                    "url": "https://www.39.net/",
                    "allowed_domains": ["39.net"],
                    "start_urls": [
                        "https://disease.39.net/bjk/xiaohuaxi/",
                        "https://disease.39.net/bjk/kouqiang/"
                    ],
                    "article_pattern": "li[class*='item']",
                    "title_pattern": "h3[class*='title']",
                    "content_pattern": "div[class*='article-cont']",
                    "next_page_pattern": "a[class*='next']",
                    "max_pages": 40
                }
            ],
            "crawler": {
                "delay": [1, 3],  # 随机延迟范围（秒）
                "timeout": 20,     # 请求超时时间（秒）
                "max_retry": 3,    # 最大重试次数
                "max_workers": 5,  # 最大线程数
                "respect_robots": True,  # 是否遵守robots.txt
                "use_proxies": False     # 是否使用代理
            },
            "proxies": [],  # 代理列表
            "storage": {
                "data_dir": "crawled_data",
                "file_format": "json"  # json, csv
            },
            "filters": {
                "min_content_length": 200,  # 最小内容长度
                "must_contain_keywords": True,  # 是否必须包含关键词
                "must_contain_count": 2  # 必须包含的关键词数量
            }
        }
        
        # 如果提供了配置文件，则加载并合并
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                
                # 合并配置
                # sites合并
                if "sites" in user_config:
                    default_config["sites"] = user_config["sites"]
                
                # crawler合并
                if "crawler" in user_config:
                    for key, value in user_config["crawler"].items():
                        default_config["crawler"][key] = value
                
                # storage合并
                if "storage" in user_config:
                    for key, value in user_config["storage"].items():
                        default_config["storage"][key] = value
                
                # filters合并
                if "filters" in user_config:
                    for key, value in user_config["filters"].items():
                        default_config["filters"][key] = value
                
                # proxies替换
                if "proxies" in user_config:
                    default_config["proxies"] = user_config["proxies"]
                
                logger.info(f"已加载配置文件: {config_path}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}，使用默认配置")
        
        return default_config
    
    def _load_proxies(self) -> List[Dict[str, str]]:
        """
        加载代理IP列表
        
        Returns:
            代理列表
        """
        if not self.config["crawler"]["use_proxies"]:
            return []
        
        proxies = self.config.get("proxies", [])
        if not proxies:
            logger.warning("未配置代理IP")
        else:
            logger.info(f"已加载{len(proxies)}个代理IP")
        
        return proxies
    
    def _get_random_headers(self) -> Dict[str, str]:
        """
        生成随机请求头
        
        Returns:
            请求头字典
        """
        try:
            user_agent = self.ua.random
        except:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
            ]
            user_agent = random.choice(user_agents)
        
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        return headers
    
    def _get_random_proxy(self) -> Optional[Dict[str, str]]:
        """
        获取随机代理
        
        Returns:
            代理配置或None
        """
        if not self.proxies:
            return None
        
        return random.choice(self.proxies)
    
    def _is_allowed_by_robots(self, url: str) -> bool:
        """
        检查URL是否被robots.txt允许访问
        
        Args:
            url: 要检查的URL
            
        Returns:
            是否允许访问
        """
        if not self.respect_robots:
            return True
        
        # 解析域名
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        
        # 如果域名已在禁止列表中，则不允许访问
        if domain in self.disallowed_sites:
            return False
        
        try:
            # 获取robots.txt
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            response = requests.get(
                robots_url, 
                timeout=self.timeout, 
                headers=self._get_random_headers()
            )
            
            if response.status_code != 200:
                return True
            
            # 简单解析robots.txt
            content = response.text.lower()
            path = parsed_url.path
            
            # 检查是否有明确的Disallow指令
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("disallow:"):
                    disallow_path = line.split(":", 1)[1].strip()
                    # 检查路径是否匹配
                    if disallow_path and path.startswith(disallow_path):
                        self.disallowed_sites.add(domain)
                        return False
            
            return True
        except Exception as e:
            logger.warning(f"检查robots.txt失败: {e}，默认允许访问")
            return True
    
    def make_request(self, url: str, method: str = "GET", data: Optional[Dict] = None, 
                    retry: int = 0) -> Optional[requests.Response]:
        """
        发送HTTP请求，带有重试机制
        
        Args:
            url: 请求URL
            method: 请求方法
            data: 请求数据
            retry: 当前重试次数
            
        Returns:
            响应对象或None(失败时)
        """
        if retry > self.max_retry:
            logger.error(f"已达到最大重试次数，放弃请求: {url}")
            return None
        
        # 检查是否允许访问
        if not self._is_allowed_by_robots(url):
            logger.info(f"根据robots.txt，禁止访问: {url}")
            return None
        
        # 随机延迟
        delay = random.uniform(self.delay[0], self.delay[1])
        time.sleep(delay)
        
        # 准备请求参数
        headers = self._get_random_headers()
        proxies = self._get_random_proxy()
        
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=self.timeout,
                    verify=False
                )
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    proxies=proxies,
                    data=data,
                    timeout=self.timeout,
                    verify=False
                )
            
            # 检查响应状态
            if response.status_code != 200:
                logger.warning(f"请求失败，状态码: {response.status_code}，URL: {url}")
                # 特殊处理某些常见错误
                if response.status_code in [403, 429]:
                    logger.warning(f"可能被限流或封禁，等待时间翻倍后重试")
                    time.sleep(delay * 2)
                
                # 重试
                return self.make_request(url, method, data, retry+1)
            
            return response
        
        except requests.exceptions.Timeout:
            logger.warning(f"请求超时: {url}，进行第{retry+1}次重试")
            return self.make_request(url, method, data, retry+1)
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"连接错误: {url}，进行第{retry+1}次重试")
            # 连接错误通常需要更长的等待时间
            time.sleep(delay * 3)
            return self.make_request(url, method, data, retry+1)
        
        except Exception as e:
            logger.error(f"请求异常: {url}, {e}，进行第{retry+1}次重试")
            return self.make_request(url, method, data, retry+1)
    
    def extract_links(self, html: str, base_url: str, site_config: Dict[str, Any]) -> List[str]:
        """
        从HTML中提取链接
        
        Args:
            html: HTML内容
            base_url: 基础URL
            site_config: 站点配置
            
        Returns:
            链接列表
        """
        soup = BeautifulSoup(html, "lxml")
        links = []
        
        # 提取所有链接
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # 处理相对URL
            if href.startswith("/"):
                parsed_url = urllib.parse.urlparse(base_url)
                href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
            elif not href.startswith(("http://", "https://")):
                href = urllib.parse.urljoin(base_url, href)
            
            # 检查域名是否在允许列表中
            parsed_href = urllib.parse.urlparse(href)
            if any(domain in parsed_href.netloc for domain in site_config["allowed_domains"]):
                links.append(href)
        
        return links
    
    def extract_article_info(self, url: str, html: str, site_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        从文章页面提取信息
        
        Args:
            url: 文章URL
            html: HTML内容
            site_config: 站点配置
            
        Returns:
            文章信息字典或None(提取失败时)
        """
        soup = BeautifulSoup(html, "lxml")
        
        # 尝试提取标题
        title = None
        title_pattern = site_config.get("title_pattern")
        if title_pattern:
            title_elem = soup.select_one(title_pattern)
            if title_elem:
                title = title_elem.get_text(strip=True)
        
        # 如果没有找到标题，尝试从<title>标签提取
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
                # 清理站点名称
                for site_name in [site_config["name"], "健康", "医学", "-"]:
                    title = title.replace(site_name, "").strip()
        
        # 如果仍然没有标题，使用URL的最后一部分
        if not title:
            path = urllib.parse.urlparse(url).path
            title = os.path.basename(path).split(".")[0]
        
        # 提取内容
        content = ""
        content_pattern = site_config.get("content_pattern")
        if content_pattern:
            content_elem = soup.select_one(content_pattern)
            if content_elem:
                # 移除脚本和样式元素
                for script in content_elem(["script", "style"]):
                    script.decompose()
                
                content = content_elem.get_text("\n", strip=True)
        
        # 如果没有找到内容，尝试使用<article>或主要<div>
        if not content:
            article = soup.find("article")
            if article:
                content = article.get_text("\n", strip=True)
            else:
                # 查找最长的div
                divs = soup.find_all("div")
                if divs:
                    longest_div = max(divs, key=lambda d: len(d.get_text()))
                    content = longest_div.get_text("\n", strip=True)
        
        # 检查内容是否满足最小长度要求
        min_length = self.config["filters"]["min_content_length"]
        if len(content) < min_length:
            logger.debug(f"内容长度不足({len(content)} < {min_length}): {url}")
            return None
        
        # 检查是否包含消化内科关键词
        if self.config["filters"]["must_contain_keywords"]:
            min_keyword_count = self.config["filters"]["must_contain_count"]
            keyword_count = sum(1 for kw in GASTRO_KEYWORDS if kw in content)
            
            if keyword_count < min_keyword_count:
                logger.debug(f"关键词数量不足({keyword_count} < {min_keyword_count}): {url}")
                return None
        
        # 提取发布时间
        publish_time = None
        time_patterns = [
            r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?",
            r"\d{1,2}[-/月]\d{1,2}[-/日号]?\s*,?\s*\d{4}"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, html)
            if match:
                publish_time = match.group(0)
                break
        
        # 提取作者
        author = None
        author_patterns = [
            r"作者[：:]\s*([^<>\s]+)",
            r"编辑[：:]\s*([^<>\s]+)",
            r"来源[：:]\s*([^<>\s]+)"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, html)
            if match:
                author = match.group(1)
                break
        
        # 提取疾病名称
        diseases = []
        for disease in GASTRO_KEYWORDS:
            if disease in content:
                diseases.append(disease)
        
        # 生成文档ID
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # 构建文章信息
        article_info = {
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "author": author,
            "site": site_config["name"],
            "crawl_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diseases": diseases
        }
        
        return article_info
    
    def save_data(self, data: List[Dict[str, Any]], site_name: str) -> None:
        """
        保存爬取的数据
        
        Args:
            data: 数据列表
            site_name: 站点名称
        """
        if not data:
            logger.warning(f"没有数据需要保存: {site_name}")
            return
        
        # 创建站点目录
        site_dir = self.data_dir / site_name
        site_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{site_name}_{timestamp}"
        
        # 根据配置的格式保存
        if self.file_format.lower() == "json":
            file_path = site_dir / f"{filename}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif self.file_format.lower() == "csv":
            file_path = site_dir / f"{filename}.csv"
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        else:
            logger.error(f"不支持的文件格式: {self.file_format}")
            return
        
        logger.info(f"已保存 {len(data)} 条记录到 {file_path}")
    
    def save_crawled_urls(self) -> None:
        """保存已爬取的URL列表"""
        try:
            with open(self.crawled_urls_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.crawled_urls), f)
            logger.info(f"已保存 {len(self.crawled_urls)} 个爬取过的URL")
        except Exception as e:
            logger.error(f"保存已爬取URL失败: {e}")
    
    def crawl_page(self, url: str, site_config: Dict[str, Any]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """
        爬取单个页面
        
        Args:
            url: 页面URL
            site_config: 站点配置
            
        Returns:
            (链接列表, 文章信息)元组
        """
        # 检查URL是否已爬取
        if url in self.crawled_urls:
            return [], None
        
        # 发送请求
        response = self.make_request(url)
        if not response:
            self.stats["failed_crawls"] += 1
            return [], None
        
        # 更新统计信息
        self.stats["total_urls"] += 1
        self.stats["successful_crawls"] += 1
        
        # 记录已爬取
        self.crawled_urls.add(url)
        
        # 提取链接
        try:
            links = self.extract_links(response.text, url, site_config)
        except Exception as e:
            logger.error(f"提取链接失败: {url}, {e}")
            links = []
        
        # 尝试提取文章信息
        article_info = None
        try:
            article_info = self.extract_article_info(url, response.text, site_config)
            if article_info:
                self.stats["extracted_data"] += 1
        except Exception as e:
            logger.error(f"提取文章信息失败: {url}, {e}")
        
        return links, article_info
    
    def crawl_site(self, site_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        爬取单个站点
        
        Args:
            site_config: 站点配置
            
        Returns:
            爬取的文章信息列表
        """
        site_name = site_config["name"]
        start_urls = site_config["start_urls"]
        max_pages = site_config.get("max_pages", 30)
        
        logger.info(f"开始爬取站点: {site_name}")
        
        # 待爬取URL队列
        to_crawl = list(start_urls)
        # 已爬取URL
        crawled = set()
        # 爬取到的文章
        articles = []
        # 计数器
        page_count = 0
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while to_crawl and page_count < max_pages:
                # 获取要爬取的URL批次
                batch_size = min(len(to_crawl), self.max_workers)
                batch_urls = to_crawl[:batch_size]
                to_crawl = to_crawl[batch_size:]
                
                # 提交爬取任务
                future_to_url = {
                    executor.submit(self.crawl_page, url, site_config): url
                    for url in batch_urls if url not in crawled
                }
                
                # 处理结果
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    crawled.add(url)
                    
                    try:
                        new_links, article_info = future.result()
                        
                        # 添加新链接到待爬取队列
                        for link in new_links:
                            if link not in crawled and link not in to_crawl:
                                to_crawl.append(link)
                        
                        # 添加文章信息
                        if article_info:
                            articles.append(article_info)
                    
                    except Exception as e:
                        logger.error(f"处理爬取结果异常: {url}, {e}")
                
                # 更新计数器
                page_count += batch_size
                
                # 定期保存数据
                if len(articles) >= 50:
                    self.save_data(articles, site_name)
                    articles = []
                
                logger.info(f"站点 {site_name} 已爬取 {page_count}/{max_pages} 页，找到 {len(crawled)} 个链接")
                
                # 防止过快爬取
                time.sleep(random.uniform(1, 2))
        
        # 保存剩余数据
        if articles:
            self.save_data(articles, site_name)
        
        logger.info(f"站点 {site_name} 爬取完成，共爬取 {len(crawled)} 个URL，提取 {len(articles)} 篇文章")
        
        return articles
    
    def run(self) -> None:
        """运行爬虫"""
        logger.info("开始爬取消化内科医疗数据")
        
        all_articles = []
        
        try:
            # 爬取每个站点
            for site_config in self.sites:
                articles = self.crawl_site(site_config)
                all_articles.extend(articles)
                
                # 站点间休息一段时间
                time.sleep(random.uniform(5, 10))
            
            # 保存全部数据
            self.save_data(all_articles, "all_sites")
            
            # 保存已爬取URL
            self.save_crawled_urls()
            
            # 打印统计信息
            self._print_stats()
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在保存数据...")
            self.save_data(all_articles, "all_sites_interrupted")
            self.save_crawled_urls()
            self._print_stats()
        
        except Exception as e:
            logger.error(f"爬虫运行异常: {e}")
            # 尝试保存已爬取的数据
            if all_articles:
                self.save_data(all_articles, "all_sites_error")
            self.save_crawled_urls()
    
    def _print_stats(self) -> None:
        """打印爬虫统计信息"""
        elapsed_time = time.time() - self.stats["start_time"]
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("\n" + "="*50)
        logger.info("爬虫统计信息:")
        logger.info(f"运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        logger.info(f"爬取URL总数: {self.stats['total_urls']}")
        logger.info(f"成功爬取数: {self.stats['successful_crawls']}")
        logger.info(f"失败爬取数: {self.stats['failed_crawls']}")
        logger.info(f"提取文章数: {self.stats['extracted_data']}")
        if self.stats['total_urls'] > 0:
            success_rate = self.stats['successful_crawls'] / self.stats['total_urls'] * 100
            logger.info(f"成功率: {success_rate:.2f}%")
        logger.info("="*50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="消化内科医疗数据爬虫")
    parser.add_argument("-c", "--config", type=str, help="配置文件路径")
    parser.add_argument("-o", "--output", type=str, help="输出目录")
    parser.add_argument("-f", "--format", choices=["json", "csv"], help="输出格式")
    parser.add_argument("-w", "--workers", type=int, help="爬虫线程数")
    parser.add_argument("-d", "--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger("MedicalCrawler").setLevel(logging.DEBUG)
    
    # 初始化爬虫
    crawler = MedicalDataCrawler(args.config)
    
    # 更新配置
    if args.output:
        crawler.data_dir = Path(args.output)
        crawler.data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format:
        crawler.file_format = args.format
    
    if args.workers:
        crawler.max_workers = args.workers
    
    # 运行爬虫
    crawler.run()

if __name__ == "__main__":
    main()