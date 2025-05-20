#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医疗问答数据集处理工具
用于清洗、转换和增强医疗问答数据，为训练做准备
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from pathlib import Path
import re
import jieba
import logging
import hashlib
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataProcessor")

class MedicalDataProcessor:
    """医疗问答数据处理器"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        medical_dict_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        max_workers: int = 4
    ):
        """
        初始化数据处理器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            medical_dict_path: 医学词典路径
            stopwords_path: 停用词表路径
            max_workers: 最大工作线程数
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载医学词典
        if medical_dict_path and os.path.exists(medical_dict_path):
            try:
                jieba.load_userdict(medical_dict_path)
                logger.info(f"已加载医学词典: {medical_dict_path}")
            except Exception as e:
                logger.error(f"加载医学词典失败: {e}")
        
        # 加载停用词
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = set([line.strip() for line in f])
                logger.info(f"已加载 {len(self.stopwords)} 个停用词")
            except Exception as e:
                logger.error(f"加载停用词失败: {e}")
        
        # 统计信息
        self.stats = {
            "processed_files": 0,
            "processed_samples": 0,
            "filtered_samples": 0,
            "augmented_samples": 0
        }
    
    def load_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据列表
        """
        file_path = Path(file_path)
        data = []
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            
            elif file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
                df = pd.read_excel(file_path)
                data = df.to_dict('records')
            
            elif file_path.suffix.lower() == '.txt':
                # 假设每行是一个JSON对象
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                # 如果不是JSON，就当作普通文本处理
                                data.append({"text": line})
            
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []
            
            logger.info(f"成功加载 {len(data)} 条记录: {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,?!;:()，。？！；：（）]', '', text)
        
        # 标准化标点
        punctuation_map = {
            '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
            '（': '(', '）': ')', '"': '"', '"': '"', ''': "'", ''': "'"
        }
        for cn, en in punctuation_map.items():
            text = text.replace(cn, en)
        
        return text.strip()
    
    def normalize_qa_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        标准化问答格式
        
        Args:
            sample: 样本数据
            
        Returns:
            标准化后的问答样本或None(无效样本)
        """
        # 确定问题字段
        question = None
        for q_field in ['question', 'query', 'q', 'title', 'problem']:
            if q_field in sample and sample[q_field]:
                question = str(sample[q_field])
                break
        
        # 确定回答字段
        answer = None
        for a_field in ['answer', 'response', 'a', 'content', 'solution', 'reply']:
            if a_field in sample and sample[a_field]:
                answer = str(sample[a_field])
                break
        
        # 如果没有问题或回答，尝试从text字段解析
        if (not question or not answer) and 'text' in sample:
            text = str(sample['text'])
            # 尝试分割问题和回答
            qa_patterns = [
                r'问[:：]\s*(.*?)\s*答[:：]\s*(.*)',
                r'Q[:：]\s*(.*?)\s*A[:：]\s*(.*)',
                r'问题[:：]\s*(.*?)\s*回答[:：]\s*(.*)',
                r'(.+?)[?？]\s*(.*)'
            ]
            
            for pattern in qa_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    break
        
        # 如果仍然缺少问题或回答，则无效
        if not question or not answer:
            return None
        
        # 清理问题和回答
        question = self.clean_text(question)
        answer = self.clean_text(answer)
        
        # 检查清理后的文本是否有效
        if len(question) < 5 or len(answer) < 10:
            return None
        
        # 构建标准格式
        normalized = {
            "question": question,
            "answer": answer,
            "id": hashlib.md5((question + answer).encode()).hexdigest()
        }
        
        # 继承其他有用字段
        for field in ['category', 'source', 'tags', 'domain', 'difficulty']:
            if field in sample and sample[field]:
                normalized[field] = sample[field]
        
        # 添加医疗领域标记
        if 'domain' not in normalized:
            normalized['domain'] = 'medical'
        
        return normalized
    
    def filter_medical_qa(self, sample: Dict[str, Any]) -> bool:
        """
        过滤非医疗问答
        
        Args:
            sample: 问答样本
            
        Returns:
            是否是医疗问答
        """
        # 医疗关键词
        medical_keywords = [
            '医生', '医院', '疾病', '症状', '治疗', '手术', '药物', '药品', '检查', 
            '诊断', '患者', '病情', '病人', '感染', '炎症', '综合征', '康复', 
            '保健', '预防', '病毒', '细菌', '抗生素', '内科', '外科', '肿瘤',
            '癌症', '心脏', '肝脏', '肾脏', '肺', '胃', '肠', '胆', '胰腺',
            '脑', '血压', '血糖', '体温', '脉搏', '呼吸', '消化', '呕吐',
            '腹泻', '便秘', '腹痛', '头痛', '发热', '咳嗽', '喘息', '皮疹'
        ]
        
        # 检查问题或回答中是否包含医疗关键词
        text = sample.get('question', '') + ' ' + sample.get('answer', '')
        
        # 计算包含的医疗关键词数量
        keyword_count = sum(1 for kw in medical_keywords if kw in text)
        
        # 如果包含至少3个医疗关键词，则认为是医疗问答
        return keyword_count >= 3
    
    def augment_qa(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        数据增强
        
        Args:
            sample: 问答样本
            
        Returns:
            增强后的样本列表(包括原样本)
        """
        results = [sample]  # 包含原始样本
        
        question = sample['question']
        answer = sample['answer']
        
        # 1. 问题改写
        q_variants = self._generate_question_variants(question)
        for q_var in q_variants:
            if q_var != question:
                new_sample = sample.copy()
                new_sample['question'] = q_var
                new_sample['id'] = hashlib.md5((q_var + answer).encode()).hexdigest()
                new_sample['augmented'] = True
                new_sample['original_id'] = sample['id']
                results.append(new_sample)
        
        # 2. 回答摘要(简化版本)
        if len(answer) > 200:
            # 简单截取前150个字符作为摘要
            summary = answer[:150] + "..."
            
            new_sample = sample.copy()
            new_sample['answer'] = summary
            new_sample['id'] = hashlib.md5((question + summary).encode()).hexdigest()
            new_sample['augmented'] = True
            new_sample['original_id'] = sample['id']
            new_sample['is_summary'] = True
            results.append(new_sample)
        
        return results
    
    def _generate_question_variants(self, question: str) -> List[str]:
        """
        生成问题变体
        
        Args:
            question: 原始问题
            
        Returns:
            问题变体列表
        """
        variants = [question]  # 包含原始问题
        
        # 1. 替换问题开头
        q_prefixes = {
            "请问": ["有谁知道", "我想请教一下", "医生您好，"],
            "医生": ["请问医生", "医生您好", "专家"],
            "如何": ["怎样", "怎么", "用什么方法可以"],
            "什么是": ["请解释一下", "能告诉我", ""] 
        }
        
        for old, new_list in q_prefixes.items():
            if question.startswith(old):
                for new in new_list:
                    new_q = new + question[len(old):]
                    if new_q != question:
                        variants.append(new_q)
        
        # 2. 添加礼貌用语
        polite_suffixes = ["谢谢！", "感谢回答。", "麻烦解答一下。"]
        for suffix in polite_suffixes:
            if not question.endswith(suffix):
                variants.append(question + suffix)
        
        # 3. 改变疑问词
        replacements = [
            (r'是什么', r'是怎样的'),
            (r'如何', r'怎么'),
            (r'怎么办', r'应该怎么处理'),
            (r'有没有', r'是否有'),
            (r'需要注意什么', r'有哪些注意事项')
        ]
        
        for old, new in replacements:
            if re.search(old, question):
                new_q = re.sub(old, new, question)
                if new_q != question:
                    variants.append(new_q)
        
        return list(set(variants))
    
    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        处理单个数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            处理后的数据列表
        """
        # 加载数据
        raw_data = self.load_data(file_path)
        
        if not raw_data:
            return []
        
        processed_samples = []
        filtered_count = 0
        augmented_count = 0
        
        # 处理每个样本
        for sample in raw_data:
            # 标准化格式
            normalized = self.normalize_qa_format(sample)
            
            if not normalized:
                filtered_count += 1
                continue
            
            # 过滤非医疗问答
            if not self.filter_medical_qa(normalized):
                filtered_count += 1
                continue
            
            # 数据增强
            augmented = self.augment_qa(normalized)
            augmented_count += len(augmented) - 1  # 减去原样本
            
            processed_samples.extend(augmented)
        
        # 更新统计信息
        self.stats["processed_samples"] += len(processed_samples)
        self.stats["filtered_samples"] += filtered_count
        self.stats["augmented_samples"] += augmented_count
        self.stats["processed_files"] += 1
        
        logger.info(f"文件处理完成: {file_path.name}, 原始样本: {len(raw_data)}, "
                    f"处理后: {len(processed_samples)}, 过滤: {filtered_count}, "
                    f"增强生成: {augmented_count}")
        
        return processed_samples
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        处理所有数据文件
        
        Returns:
            处理后的数据集字典(train, validation, test)
        """
        all_data = []
        
        # 查找所有数据文件
        file_paths = []
        for ext in ['.json', '.csv', '.xlsx', '.xls', '.txt']:
            file_paths.extend(list(self.data_dir.glob(f'**/*{ext}')))
        
        logger.info(f"找到 {len(file_paths)} 个数据文件")
        
        # 多线程处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file_path): file_path 
                              for file_path in file_paths}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                               total=len(file_paths), desc="处理数据文件"):
                file_path = future_to_file[future]
                try:
                    processed_data = future.result()
                    all_data.extend(processed_data)
                except Exception as e:
                    logger.error(f"处理文件出错: {file_path}, {e}")
        
        # 去重
        unique_data = {}
        for sample in all_data:
            sample_id = sample['id']
            if sample_id not in unique_data:
                unique_data[sample_id] = sample
        
        # 转换为DataFrame
        df = pd.DataFrame(list(unique_data.values()))
        
        # 数据集拆分
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # 保存到文件
        train_df.to_csv(self.output_dir / "train.csv", index=False, encoding='utf-8-sig')
        val_df.to_csv(self.output_dir / "validation.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(self.output_dir / "test.csv", index=False, encoding='utf-8-sig')
        
        # 保存JSON格式
        train_df.to_json(self.output_dir / "train.json", orient='records', 
                         force_ascii=False, indent=2)
        val_df.to_json(self.output_dir / "validation.json", orient='records', 
                        force_ascii=False, indent=2)
        test_df.to_json(self.output_dir / "test.json", orient='records', 
                        force_ascii=False, indent=2)
        
        # 打印统计信息
        logger.info("\n" + "="*60)
        logger.info("数据处理完成")
        logger.info(f"处理的文件数: {self.stats['processed_files']}")
        logger.info(f"处理的样本数: {self.stats['processed_samples']}")
        logger.info(f"过滤的样本数: {self.stats['filtered_samples']}")
        logger.info(f"增强生成的样本数: {self.stats['augmented_samples']}")
        logger.info(f"去重后的样本数: {len(df)}")
        logger.info(f"训练集样本数: {len(train_df)}")
        logger.info(f"验证集样本数: {len(val_df)}")
        logger.info(f"测试集样本数: {len(test_df)}")
        logger.info("="*60)
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }
    
    def convert_to_chat_format(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        转换为聊天格式
        
        Args:
            df: 问答数据DataFrame
            output_path: 输出路径
        """
        chat_samples = []
        
        for _, row in df.iterrows():
            chat_sample = {
                "id": row.get('id', ''),
                "conversations": [
                    {"role": "user", "content": row['question']},
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            
            # 添加其他元数据字段
            for field in ['category', 'source', 'domain', 'tags']:
                if field in row and not pd.isna(row[field]):
                    chat_sample[field] = row[field]
            
            chat_samples.append(chat_sample)
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chat_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已转换 {len(chat_samples)} 个样本为聊天格式: {output_path}")
    
    def convert_all_to_chat_format(self) -> None:
        """转换所有数据集为聊天格式"""
        for split in ['train', 'validation', 'test']:
            input_path = self.output_dir / f"{split}.csv"
            output_path = self.output_dir / f"{split}_chat.json"
            
            if input_path.exists():
                df = pd.read_csv(input_path)
                self.convert_to_chat_format(df, output_path)
    
    def create_instruction_data(self) -> None:
        """创建指令微调格式数据"""
        # 加载数据
        train_path = self.output_dir / "train.csv"
        
        if not train_path.exists():
            logger.error(f"找不到训练数据: {train_path}")
            return
        
        train_df = pd.read_csv(train_path)
        
        # 创建指令数据
        instructions = []
        
        # 医疗问答模板
        templates = [
            "请回答以下医疗问题：\n\n{question}",
            "作为一名医疗助手，请解答下面的问题：\n\n{question}",
            "以下是一个关于健康的问题，请给出专业回答：\n\n{question}",
            "请基于医学知识回答这个问题：\n\n{question}",
            "用通俗易懂的语言回答以下医疗咨询：\n\n{question}"
        ]
        
        for _, row in train_df.iterrows():
            # 随机选择一个模板
            template = random.choice(templates)
            instruction = template.format(question=row['question'])
            
            instructions.append({
                "id": row.get('id', ''),
                "instruction": instruction,
                "input": "",  # 这里不需要额外输入
                "output": row['answer']
            })
        
        # 保存指令数据
        output_path = self.output_dir / "train_instructions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已创建 {len(instructions)} 个指令微调样本: {output_path}")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗问答数据处理工具")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--medical_dict", type=str, help="医学词典路径")
    parser.add_argument("--stopwords", type=str, help="停用词表路径")
    parser.add_argument("--workers", type=int, default=4, help="最大工作线程数")
    parser.add_argument("--chat_format", action="store_true", help="转换为聊天格式")
    parser.add_argument("--instruction_format", action="store_true", help="创建指令微调数据")
    
    args = parser.parse_args()
    
    processor = MedicalDataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        medical_dict_path=args.medical_dict,
        stopwords_path=args.stopwords,
        max_workers=args.workers
    )
    
    # 处理所有数据
    processor.process_all_data()
    
    # 可选：转换为聊天格式
    if args.chat_format:
        processor.convert_all_to_chat_format()
    
    # 可选：创建指令微调数据
    if args.instruction_format:
        processor.create_instruction_data()

















class DPODataProcessor:
    pass







class DPODataFilter:
    pass