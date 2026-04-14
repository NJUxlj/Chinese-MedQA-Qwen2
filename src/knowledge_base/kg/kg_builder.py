import os, sys
import logging
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Tuple
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings
from langchain_community.graphs.neo4j_graph import Neo4jGraph

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGTriplet(BaseModel):
    """知识图谱三元组"""
    subject: str = Field(default_factory=str)
    predicate: str = Field(default_factory=str)
    object: str = Field(default_factory=str)



class KGBuilder:
    """知识图谱构建器，用于构建知识图谱"""


    def __init__(self, config=None):
        """初始化构建器"""
        self.config = config if config is not None else settings.neo4j
        self.neo4j_graph = None
        self._setup_neo4j_connection()


    def _setup_neo4j_connection(self) -> None:
        """设置Neo4j连接"""
        try:
            # 提取数据库名称（从URI中获取或使用默认值）
            database = "neo4j"  # Neo4j的默认数据库
            
            self.neo4j_graph = Neo4jGraph(
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=database
            )
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从单个PDF文件中提取文本"""
        try:
            import pdfplumber
            import PyPDF2
            
            text_content = ""
            
            # 尝试使用pdfplumber（更好的文本提取）
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception as e:
                logger.warning(f"pdfplumber提取失败，尝试PyPDF2: {e}")
                
                # 备用方案：使用PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            
            logger.info(f"成功提取PDF文本: {pdf_path}, 长度: {len(text_content)}")
            return text_content
            
        except Exception as e:
            logger.error(f"提取PDF文本失败 {pdf_path}: {e}")
            return ""


    def preprocess_text(self, text: str) -> str:
        """预处理文本，清理和标准化"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除页码（通常单独一行的数字）
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # 移除常见的页眉页脚模式
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # 清理文本
        text = text.strip()
        
        return text


    def extract_medical_entities(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取医学实体"""
        entities = []
        
        try:
            # 使用正则表达式提取常见的医学实体
            
            # 1. 疾病和症状
            disease_patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:是一种?|是一种疾病|的症状|引起|导致|表现为)',
                r'(?:患有|得了|出现|诊断为)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:患者|病人)',
            ]
            
            for pattern in disease_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = match.group(1).strip()
                    if len(entity) > 2 and entity.isalpha():
                        entities.append({
                            'text': entity,
                            'label': 'DISEASE',
                            'start': match.start(1),
                            'end': match.end(1)
                        })
            
            # 2. 药物
            drug_patterns = [
                r'(?:药物|药|治疗|服用|使用)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:药物|药片|胶囊|注射液)',
                r'建议服用\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ]
            
            for pattern in drug_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = match.group(1).strip()
                    if len(entity) > 2 and entity.isalpha():
                        entities.append({
                            'text': entity,
                            'label': 'DRUG',
                            'start': match.start(1),
                            'end': match.end(1)
                        })
            
            # 3. 解剖结构
            anatomy_patterns = [
                r'(?:在|位于|影响)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:部位|区域|器官)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:器官|组织|系统)',
            ]
            
            for pattern in anatomy_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = match.group(1).strip()
                    if len(entity) > 2 and entity.isalpha():
                        entities.append({
                            'text': entity,
                            'label': 'ANATOMY',
                            'start': match.start(1),
                            'end': match.end(1)
                        })
            
            logger.info(f"提取到 {len(entities)} 个医学实体")
            return entities
            
        except Exception as e:
            logger.error(f"提取医学实体失败: {e}")
            return []


    def extract_relationships(self, text: str, entities: List[Dict[str, str]]) -> List[KGTriplet]:
        """从文本中提取实体关系并生成三元组"""
        triplets = []
        
        try:
            # 预定义的关系模式
            relationship_patterns = {
                'TREATS': [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+治疗\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+用于\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+可以治疗\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                ],
                'CAUSES': [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+引起\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+导致\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+造成\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                ],
                'AFFECTS': [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+影响\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+作用于\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                ],
                'LOCATED_IN': [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+位于\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+在\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+中',
                ]
            }
            
            # 为每种关系类型提取三元组
            for rel_type, patterns in relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        subject = match.group(1).strip()
                        object_entity = match.group(2).strip()
                        
                        # 验证实体是否在提取的实体列表中
                        if self._validate_entity(subject, entities) and self._validate_entity(object_entity, entities):
                            triplet = KGTriplet(
                                subject=subject,
                                predicate=rel_type,
                                object=object_entity
                            )
                            triplets.append(triplet)
            
            # 基于邻近性的简单关系推断
            triplets.extend(self._extract_proximity_relationships(text, entities))
            
            logger.info(f"提取到 {len(triplets)} 个三元组")
            return triplets
            
        except Exception as e:
            logger.error(f"提取关系失败: {e}")
            return []


    def _validate_entity(self, entity_text: str, entities: List[Dict[str, str]]) -> bool:
        """验证实体是否在已提取的实体列表中"""
        for entity in entities:
            if entity['text'].lower() == entity_text.lower():
                return True
        return False


    def _extract_proximity_relationships(self, text: str, entities: List[Dict[str, str]]) -> List[KGTriplet]:
        """基于邻近性提取关系"""
        triplets = []
        
        try:
            # 按位置排序实体
            sorted_entities = sorted(entities, key=lambda x: x['start'])
            
            # 查找相邻的实体
            for i in range(len(sorted_entities) - 1):
                current_entity = sorted_entities[i]
                next_entity = sorted_entities[i + 1]
                
                # 检查实体间的距离
                distance = next_entity['start'] - current_entity['end']
                
                # 如果实体在合理的距离内（50个字符），创建关联关系
                if distance < 50:
                    triplet = KGTriplet(
                        subject=current_entity['text'],
                        predicate='RELATED_TO',
                        object=next_entity['text']
                    )
                    triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            logger.error(f"提取邻近关系失败: {e}")
            return []



    def get_triplets_from_pdfs(self, pdf_dir: str) -> List[KGTriplet]:
        """从 PDF 目录中提取三元组"""
        all_triplets = []
        pdf_dir_path = Path(pdf_dir)
        
        if not pdf_dir_path.exists():
            logger.error(f"PDF目录不存在: {pdf_dir}")
            return []
        
        # 查找所有PDF文件
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
        
        for pdf_file in pdf_files:
            logger.info(f"处理PDF文件: {pdf_file.name}")
            
            # 提取PDF文本
            raw_text = self.extract_text_from_pdf(str(pdf_file))
            
            if not raw_text:
                logger.warning(f"PDF文件 {pdf_file.name} 无文本内容")
                continue
            
            # 预处理文本
            clean_text = self.preprocess_text(raw_text)
            
            if not clean_text:
                logger.warning(f"PDF文件 {pdf_file.name} 预处理后无有效文本")
                continue
            
            # 提取医学实体
            entities = self.extract_medical_entities(clean_text)
            
            if not entities:
                logger.warning(f"PDF文件 {pdf_file.name} 中未提取到医学实体")
                continue
            
            # 提取关系并生成三元组
            triplets = self.extract_relationships(clean_text, entities)
            
            # 添加文档来源信息
            for triplet in triplets:
                triplet.subject = f"{triplet.subject} [来源:{pdf_file.name}]"
            
            all_triplets.extend(triplets)
            logger.info(f"从 {pdf_file.name} 提取到 {len(triplets)} 个三元组")
        
        logger.info(f"总共提取到 {len(all_triplets)} 个三元组")
        return all_triplets


    def create_empty_kg(self) -> None:
        """在 neo4j 中创建一个新的知识图谱"""
        try:
            # 清空现有数据
            self.neo4j_graph.query("MATCH (n) DETACH DELETE n")
            
            # 创建索引以提高查询性能
            self.neo4j_graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)")
            self.neo4j_graph.query("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            
            # 创建约束
            try:
                self.neo4j_graph.query("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
            except Exception:
                # 约束可能已存在，忽略错误
                pass
            
            logger.info("成功创建空知识图谱")
            
        except Exception as e:
            logger.error(f"创建空知识图谱失败: {e}")
            raise


    def insert_triplets_into_kg(self, triplets: List[KGTriplet]) -> None:
        """将三元组插入到知识图谱中"""
        if not triplets:
            logger.warning("没有三元组需要插入")
            return
        
        try:
            batch_size = 100  # 批量插入以提高性能
            
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                self._insert_batch_triplets(batch)
                
            logger.info(f"成功插入 {len(triplets)} 个三元组到知识图谱")
            
        except Exception as e:
            logger.error(f"插入三元组失败: {e}")
            raise


    def _insert_batch_triplets(self, triplets: List[KGTriplet]) -> None:
        """批量插入三元组"""
        try:
            # 构建批量插入的Cypher查询
            query = """
            UNWIND $triplets AS triplet
            MERGE (subject:Entity {id: triplet.subject})
            MERGE (object:Entity {id: triplet.object})
            MERGE (subject)-[rel:RELATIONSHIP {type: triplet.predicate}]->(object)
            RETURN count(*) as inserted_count
            """
            
            # 准备数据
            triplet_data = [
                {
                    "subject": triplet.subject,
                    "predicate": triplet.predicate,
                    "object": triplet.object
                }
                for triplet in triplets
            ]
            
            # 执行批量插入
            result = self.neo4j_graph.query(query, {"triplets": triplet_data})
            
            logger.debug(f"批次插入完成，插入数量: {result[0]['inserted_count'] if result else 0}")
            
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            raise


    def build_kg_from_pdfs(self, pdf_dir: str) -> None:
        """从 PDF 目录中构建知识图谱"""
        try:
            logger.info(f"开始从PDF目录构建知识图谱: {pdf_dir}")
            
            # 第一步：从PDF提取三元组
            logger.info("步骤1: 从PDF提取三元组...")
            triplets = self.get_triplets_from_pdfs(pdf_dir)
            
            if not triplets:
                logger.warning("未提取到任何三元组，构建过程终止")
                return
            
            # 第二步：创建空知识图谱
            logger.info("步骤2: 创建空知识图谱...")
            self.create_empty_kg()
            
            # 第三步：插入三元组
            logger.info("步骤3: 插入三元组到知识图谱...")
            self.insert_triplets_into_kg(triplets)
            
            # 第四步：生成统计信息
            self._print_kg_statistics()
            
            logger.info("知识图谱构建完成!")
            
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            raise


    def _print_kg_statistics(self) -> None:
        """打印知识图谱统计信息"""
        try:
            # 获取节点数量
            node_count_result = self.neo4j_graph.query("MATCH (n:Entity) RETURN count(n) as node_count")
            node_count = node_count_result[0]['node_count'] if node_count_result else 0
            
            # 获取关系数量
            rel_count_result = self.neo4j_graph.query("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as rel_count")
            rel_count = rel_count_result[0]['rel_count'] if rel_count_result else 0
            
            # 获取关系类型统计
            rel_types_result = self.neo4j_graph.query("""
                MATCH ()-[r:RELATIONSHIP]->() 
                RETURN r.type as relationship_type, count(*) as count
                ORDER BY count DESC
            """)
            
            logger.info("="*50)
            logger.info("知识图谱统计信息:")
            logger.info(f"节点数量: {node_count}")
            logger.info(f"关系数量: {rel_count}")
            logger.info("关系类型分布:")
            for rel_type_info in rel_types_result:
                logger.info(f"  {rel_type_info['relationship_type']}: {rel_type_info['count']}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")


    def get_kg_summary(self) -> Dict[str, Any]:
        """获取知识图谱摘要信息"""
        try:
            summary = {}
            
            # 节点统计
            node_count_result = self.neo4j_graph.query("MATCH (n:Entity) RETURN count(n) as count")
            summary['node_count'] = node_count_result[0]['count'] if node_count_result else 0
            
            # 关系统计
            rel_count_result = self.neo4j_graph.query("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as count")
            summary['relationship_count'] = rel_count_result[0]['count'] if rel_count_result else 0
            
            # 关系类型分布
            rel_types_result = self.neo4j_graph.query("""
                MATCH ()-[r:RELATIONSHIP]->() 
                RETURN r.type as type, count(*) as count
                ORDER BY count DESC
            """)
            summary['relationship_types'] = {item['type']: item['count'] for item in rel_types_result}
            
            # 示例三元组
            sample_result = self.neo4j_graph.query("""
                MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
                RETURN s.id as subject, r.type as predicate, o.id as object
                LIMIT 5
            """)
            summary['sample_triplets'] = [
                {'subject': item['subject'], 'predicate': item['predicate'], 'object': item['object']}
                for item in sample_result
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"获取知识图谱摘要失败: {e}")
            return {}


    def close_connection(self) -> None:
        """关闭Neo4j连接"""
        if self.neo4j_graph and self.neo4j_graph.driver:
            self.neo4j_graph.driver.close()
            logger.info("Neo4j连接已关闭")





if __name__ == "__main__":
    kg_builder = KGBuilder()
    kg_builder.build_kg_from_pdfs("data/pdfs")



    