#!/usr/bin/env python3
"""
知识图谱构建器测试脚本
用于测试医疗PDF文档知识图谱构建的各个组件
"""

import os
import sys
import unittest
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 添加src目录到Python路径
src_path = project_root / "src"
sys.path.append(str(src_path))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目模块
try:
    from config.kg_config import KGConfig
    from knowledge_base.kg.kg_builder import KGBuilder, KGTriplet
    from knowledge_base.pdf.pdf_parser import PDFParser
    from langchain_community.graphs.neo4j_graph import Neo4jGraph
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class TestKGBuilder(unittest.TestCase):
    """知识图谱构建器测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logger.info("开始测试知识图谱构建器...")
        
        # 创建临时配置文件
        cls.temp_dir = tempfile.mkdtemp()
        cls.config_file = os.path.join(cls.temp_dir, "test_config.json")
        
        # 模拟配置数据
        cls.test_config_data = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j", 
            "password": "password",
            "database": "medical_kg"
        }
        
        with open(cls.config_file, 'w') as f:
            json.dump(cls.test_config_data, f)
        
        # 创建测试配置
        cls.config = KGConfig(
            uri=cls.test_config_data["uri"],
            username=cls.test_config_data["username"],
            password=cls.test_config_data["password"]
        )

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """测试方法初始化"""
        # 模拟Neo4j连接
        self.mock_neo4j = Mock(spec=Neo4jGraph)
        
        # 创建测试用的KGBuilder实例
        with patch('knowledge_base.kg.kg_builder.Neo4jGraph') as mock_neo4j_class:
            mock_neo4j_class.return_value = self.mock_neo4j
            self.kg_builder = KGBuilder(self.config)

    def test_config_loading(self):
        """测试配置加载"""
        logger.info("测试配置加载...")
        
        self.assertEqual(self.config.uri, "bolt://localhost:7687")
        self.assertEqual(self.config.username, "neo4j")
        self.assertEqual(self.config.password, "password")
        logger.info("✓ 配置加载测试通过")

    def test_kg_triplet_creation(self):
        """测试三元组创建"""
        logger.info("测试三元组创建...")
        
        triplet = KGTriplet(
            subject="高血压",
            predicate="导致", 
            object="心血管疾病"
        )
        
        self.assertEqual(triplet.subject, "高血压")
        self.assertEqual(triplet.predicate, "导致")
        self.assertEqual(triplet.object, "心血管疾病")
        logger.info("✓ 三元组创建测试通过")

    def test_medical_entity_extraction(self):
        """测试医疗实体提取"""
        logger.info("测试医疗实体提取...")
        
        test_text = """
        高血压是一种常见的慢性疾病，患者常伴有头痛、头晕等症状。
        治疗高血压常用的药物包括ACE抑制剂和钙通道阻滞剂。
        该疾病可能导致心血管疾病和肾脏疾病。
        """
        
        # 测试实体提取方法（需要实际实现）
        entities = self.kg_builder.extract_entities_from_text(test_text)
        
        # 验证提取的实体
        self.assertIsInstance(entities, dict)
        self.assertIn('disease', entities)
        self.assertIn('drug', entities)
        self.assertIn('symptom', entities)
        
        # 检查是否包含预期的实体
        disease_entities = entities.get('disease', [])
        self.assertTrue(any('高血压' in disease for disease in disease_entities))
        
        logger.info(f"提取的实体: {entities}")
        logger.info("✓ 医疗实体提取测试通过")

    def test_relationship_extraction(self):
        """测试关系提取"""
        logger.info("测试关系提取...")
        
        test_text = """
        高血压导致心血管疾病。
        ACE抑制剂用于治疗高血压。
        头痛是高血压的常见症状。
        """
        
        # 测试关系提取方法
        relationships = self.kg_builder.extract_relationships(test_text)
        
        self.assertIsInstance(relationships, list)
        
        # 验证关系结构
        for relationship in relationships:
            self.assertIn('subject', relationship)
            self.assertIn('predicate', relationship)
            self.assertIn('object', relationship)
            self.assertIn('confidence', relationship)
        
        logger.info(f"提取的关系: {relationships}")
        logger.info("✓ 关系提取测试通过")

    def test_triplet_generation(self):
        """测试三元组生成"""
        logger.info("测试三元组生成...")
        
        test_text = """
        糖尿病是一种代谢性疾病，特征是高血糖。
        胰岛素用于治疗糖尿病。
        糖尿病并发症包括视网膜病变和肾病。
        """
        
        # 测试三元组生成
        triplets = self.kg_builder.generate_triplets_from_text(test_text)
        
        self.assertIsInstance(triplets, list)
        self.assertTrue(len(triplets) > 0)
        
        # 验证三元组结构
        for triplet in triplets:
            self.assertIsInstance(triplet, KGTriplet)
            self.assertTrue(hasattr(triplet, 'subject'))
            self.assertTrue(hasattr(triplet, 'predicate'))
            self.assertTrue(hasattr(triplet, 'object'))
        
        logger.info(f"生成的三元组: {[t.dict() for t in triplets]}")
        logger.info("✓ 三元组生成测试通过")

    def test_neo4j_operations(self):
        """测试Neo4j操作"""
        logger.info("测试Neo4j操作...")
        
        # 模拟Neo4j操作
        self.mock_neo4j.query.return_value = {"nodes": 10, "relationships": 25}
        
        # 测试创建空图谱
        self.kg_builder.create_empty_kg()
        self.mock_neo4j.query.assert_called()
        
        # 测试插入三元组
        test_triplets = [
            KGTriplet(subject="疾病A", predicate="导致", object="症状B"),
            KGTriplet(subject="药物C", predicate="治疗", object="疾病A")
        ]
        
        self.kg_builder.insert_triplets_into_kg(test_triplets)
        
        # 验证调用次数（每个三元组应该调用一次）
        self.assertEqual(self.mock_neo4j.query.call_count, 3)  # 1次创建 + 2次插入
        
        logger.info("✓ Neo4j操作测试通过")

    @patch('knowledge_base.pdf.pdf_parser.PDFParser')
    def test_pdf_processing(self, mock_pdf_parser):
        """测试PDF处理"""
        logger.info("测试PDF处理...")
        
        # 模拟PDF解析器
        mock_parser_instance = Mock()
        mock_parser_instance.parse_to_documents.return_value = [
            Mock(page_content="测试医疗文本内容", metadata={"filename": "test.pdf"})
        ]
        mock_pdf_parser.return_value = mock_parser_instance
        
        # 创建PDF解析器实例
        pdf_parser = PDFParser()
        
        # 测试文档解析
        documents = pdf_parser.parse_to_documents("test.pdf")
        
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "测试医疗文本内容")
        self.assertEqual(documents[0].metadata["filename"], "test.pdf")
        
        logger.info("✓ PDF处理测试通过")

    def test_batch_pdf_processing(self):
        """测试批量PDF处理"""
        logger.info("测试批量PDF处理...")
        
        # 模拟多个PDF文件
        mock_documents = [
            Mock(page_content="文本1", metadata={"filename": "doc1.pdf"}),
            Mock(page_content="文本2", metadata={"filename": "doc2.pdf"})
        ]
        
        with patch.object(self.kg_builder, 'extract_entities_from_text') as mock_extract:
            mock_extract.return_value = {'disease': ['糖尿病'], 'drug': ['胰岛素']}
            
            with patch.object(self.kg_builder, 'extract_relationships') as mock_rel:
                mock_rel.return_value = [
                    {'subject': '糖尿病', 'predicate': '导致', 'object': '高血糖', 'confidence': 0.9}
                ]
                
                # 测试批量处理
                triplets = self.kg_builder.get_triplets_from_pdfs("mock_pdf_dir")
                
                self.assertIsInstance(triplets, list)
                logger.info(f"批量处理生成的三元组: {triplets}")
        
        logger.info("✓ 批量PDF处理测试通过")

    def test_kg_statistics(self):
        """测试知识图谱统计"""
        logger.info("测试知识图谱统计...")
        
        # 模拟Neo4j查询结果
        self.mock_neo4j.query.return_value = {
            "nodes": [
                {"labels": ["Entity"], "properties": {"type": "disease"}},
                {"labels": ["Entity"], "properties": {"type": "drug"}},
                {"labels": ["Entity"], "properties": {"type": "symptom"}}
            ],
            "relationships": [
                {"type": "RELATION", "properties": {"type": "导致"}},
                {"type": "RELATION", "properties": {"type": "治疗"}}
            ]
        }
        
        # 测试统计信息生成
        with patch('builtins.print') as mock_print:
            self.kg_builder._print_kg_statistics()
            
            # 验证打印调用
            self.assertTrue(mock_print.called)
        
        logger.info("✓ 知识图谱统计测试通过")

    def test_error_handling(self):
        """测试错误处理"""
        logger.info("测试错误处理...")
        
        # 测试空文本处理
        empty_result = self.kg_builder.generate_triplets_from_text("")
        self.assertEqual(empty_result, [])
        
        # 测试无效PDF路径
        with patch.object(self.kg_builder, 'get_triplets_from_pdfs') as mock_get:
            mock_get.side_effect = FileNotFoundError("文件不存在")
            
            with self.assertRaises(FileNotFoundError):
                self.kg_builder.build_kg_from_pdfs("invalid_path")
        
        logger.info("✓ 错误处理测试通过")

    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        logger.info("测试端到端工作流...")
        
        # 模拟完整的端到端流程
        test_pdf_dir = "test_pdfs"
        
        # 模拟各个组件的返回值
        mock_triplets = [
            KGTriplet(subject="疾病X", predicate="导致", object="症状Y"),
            KGTriplet(subject="药物Z", predicate="治疗", object="疾病X")
        ]
        
        with patch.object(self.kg_builder, 'get_triplets_from_pdfs') as mock_get:
            mock_get.return_value = mock_triplets
            
            with patch.object(self.kg_builder, 'create_empty_kg') as mock_create:
                with patch.object(self.kg_builder, 'insert_triplets_into_kg') as mock_insert:
                    with patch.object(self.kg_builder, '_print_kg_statistics') as mock_stats:
                        
                        # 执行端到端流程
                        self.kg_builder.build_kg_from_pdfs(test_pdf_dir)
                        
                        # 验证各个方法被调用
                        mock_get.assert_called_once_with(test_pdf_dir)
                        mock_create.assert_called_once()
                        mock_insert.assert_called_once_with(mock_triplets)
                        mock_stats.assert_called_once()
        
        logger.info("✓ 端到端工作流测试通过")


class TestPDFParser(unittest.TestCase):
    """PDF解析器测试类"""

    def setUp(self):
        """测试初始化"""
        self.pdf_parser = PDFParser()

    def test_supported_formats(self):
        """测试支持的格式"""
        self.assertIn('.pdf', self.pdf_parser.supported_formats)
        logger.info("✓ 支持格式测试通过")

    def test_text_extraction_fallback(self):
        """测试文本提取降级机制"""
        logger.info("测试文本提取降级机制...")
        
        # 测试空文本处理
        result = self.pdf_parser._extract_text_with_metadata("nonexistent.pdf")
        self.assertEqual(result['text'], "")
        self.assertEqual(result['method'], "failed")
        
        logger.info("✓ 文本提取降级机制测试通过")


def run_comprehensive_tests():
    """运行综合测试"""
    logger.info("=" * 60)
    logger.info("开始运行知识图谱构建器综合测试")
    logger.info("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [TestKGBuilder, TestPDFParser]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    logger.info("=" * 60)
    if result.wasSuccessful():
        logger.info("🎉 所有测试通过！知识图谱构建器功能正常。")
    else:
        logger.error(f"❌ 测试失败！失败测试数量: {len(result.failures)}, 错误测试数量: {len(result.errors)}")
        
        if result.failures:
            logger.error("失败的测试:")
            for test, traceback in result.failures:
                logger.error(f"  - {test}: {traceback}")
        
        if result.errors:
            logger.error("错误的测试:")
            for test, traceback in result.errors:
                logger.error(f"  - {test}: {traceback}")
    
    logger.info("=" * 60)
    
    return result.wasSuccessful()


def create_test_sample_data():
    """创建测试样本数据"""
    logger.info("创建测试样本数据...")
    
    # 创建临时测试目录
    test_dir = Path(project_root) / "tests" / "sample_data"
    test_dir.mkdir(exist_ok=True)
    
    # 创建模拟医疗文本文件
    medical_text = """
    医疗知识文本样本
    
    疾病概述:
    高血压是一种常见的慢性疾病，主要特征是动脉血压持续升高。
    该疾病可分为原发性高血压和继发性高血压两种类型。
    
    临床症状:
    高血压患者常伴有头痛、头晕、耳鸣等症状。
    严重时可能出现视力模糊、胸闷、心悸等表现。
    
    治疗方案:
    治疗高血压的常用药物包括ACE抑制剂、钙通道阻滞剂、利尿剂等。
    ACE抑制剂通过抑制血管紧张素转换酶来降低血压。
    钙通道阻滞剂通过阻断钙离子通道来扩张血管。
    
    并发症:
    长期高血压可能导致心血管疾病、脑卒中、肾脏疾病等严重并发症。
    心血管疾病是高血压最主要的并发症之一。
    
    预防措施:
    预防高血压需要控制体重、减少盐分摄入、适量运动、戒烟限酒。
    定期监测血压，及时发现和治疗非常重要。
    """
    
    sample_file = test_dir / "medical_sample.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(medical_text)
    
    logger.info(f"测试样本数据已创建: {sample_file}")
    return str(sample_file)


if __name__ == "__main__":
    try:
        # 创建测试样本数据
        sample_data_path = create_test_sample_data()
        
        # 运行综合测试
        success = run_comprehensive_tests()
        
        if success:
            logger.info("\n🚀 知识图谱构建器测试完成！系统准备就绪。")
            logger.info(f"📁 测试样本数据位置: {sample_data_path}")
        else:
            logger.error("\n⚠️  测试发现问题，请检查相关组件。")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)