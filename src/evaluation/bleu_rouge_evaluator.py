import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))  
import numpy as np  
from torch.fx.experimental.symbolic_shapes import InputList
from tqdm import tqdm  
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  
from torch.utils.data import DataLoader, Dataset
import evaluate  

from typing import Dict, List, Any


from config.evaluator_config import BleuRougeEvaluatorConfig
from evaluation.base_evaluator import BaseEvaluator, EvaluatorDataset



class BleuRougeEvaluator(BaseEvaluator):  
    def __init__(self, config: BleuRougeEvaluatorConfig):  
        """  
        初始化评估器  
        :param model_path: 微调后的模型路径  
        :param tokenizer_path: 分词器路径  
        :param device: 计算设备 (default: cuda:0)  
        """  
        super().__init__(config)
        
        # 加载模型和分词器  
        self.model = AutoModelForCausalLM.from_pretrained(  
            self.config.model_name_or_path,  
            trust_remote_code=True,  
            torch_dtype=torch.bfloat16  
        ).to(self.config.device)  


        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,  
            trust_remote_code=True  
        )


        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # 初始化评估指标  
        self.bleu = evaluate.load("bleu")  
        self.rouge = evaluate.load("rouge")  
        
        
        # 生成参数配置  
        self.generation_config = {  
            "max_new_tokens": 2048,  
            "temperature": 0.7,  
            "top_p": 0.9,  
            "do_sample": True,  
            "pad_token_id": self.tokenizer.eos_token_id  
        }  

        self.logger.info(f"BLEU-ROUGE评估器初始化完成，模型路径: {self.config.model_name_or_path}")

    
    def evaluate_one_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        return None
        
    
    def evaluate_batch_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估数据集
        """
        results = {
            "perplexity": [],  
            "bleu": [],  
            "rouge": [],  
        }

        prompts = [sample[self.config.prompt_key] for sample in samples]
        ground_true_answers = [sample[self.config.ground_true_answer_key] for sample in samples]

        inputs = self.tokenizer(  
            prompts,  
            return_tensors="pt",  
            padding=True,  
            truncation=True,  
            max_length=1024  
        ).to(self.config.device)  
        
        # 生成文本  
        outputs = self.model.generate(  
            **inputs,  
            **self.generation_config  
        )  
        predictions = self.tokenizer.batch_decode(  
            outputs[:, inputs["input_ids"].shape[1]:],   
            skip_special_tokens=True  
        )  
        
        # 计算困惑度  
        loss = self.model(  
            inputs["input_ids"],  
            labels=inputs["input_ids"]  
        ).loss  
        perplexity = torch.exp(loss).item()  
        results["perplexity"].append(perplexity)  

        # 计算文本相似度指标  
        references = ground_true_answers
        results["bleu"].extend([  
            self.bleu.compute(  
                predictions=[p],   
                references=[r]  
            )["bleu"] for p, r in zip(predictions, references)  
        ])  
        
        rouge_scores = self.rouge.compute(  
            predictions=predictions,  
            references=references,  
            rouge_types=["rougeL"]  
        )  
        results["rouge"].extend(rouge_scores["rougeL"])  


        return results
        
        
    
    
    def evaluate(self) -> Dict[str, float]:  
        """  
        评估数据集  
        """  
        # 加载数据  
        

        results = {  
            "perplexity": [],  
            "bleu": [],  
            "rouge": [],  
        }  

        with torch.no_grad():  
            for batch in tqdm(self.test_dataloader, desc="Evaluating"):  
                # 生成回答  
                batch_results = self.evaluate_batch_sample(batch)
                
                results["perplexity"].extend(batch_results["perplexity"])
                results["bleu"].extend(batch_results["bleu"])
                results["rouge"].extend(batch_results["rouge"])

        # 汇总结果  
        return {  
            "perplexity": np.mean(results["perplexity"]),  
            "bleu": np.mean(results["bleu"]),  
            "rougeL": np.mean(results["rouge"]),  
        }  

    def save_results(self, results: Dict[str, float]):  
        """保存评估结果"""  
        with open(self.config.eval_result_save_path, "w", encoding="utf-8") as f:  
            json.dump(results, f, ensure_ascii=False, indent=2) 
            
            


def run():
    """运行评估器"""
    config = BleuRougeEvaluatorConfig(
        model_name_or_path="/Users/xiniuyiliao/Desktop/code/models/gpt2",
        device="cpu",
        test_dataset_path="/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2/src/evaluation/data/test_data_1.json",
        prompt_key="prompt",
        ground_true_answer_key="ground_true_answer",
        per_device_eval_batch_size=2,
        eval_result_save_path="/Users/xiniuyiliao/Desktop/code/Chinese-MedQA-Qwen2/src/evaluation/eval_results.json"
    )

    evaluator = BleuRougeEvaluator(config)

    results = evaluator.evaluate()
    evaluator.save_results(results)
    print("评估完成！")
    print(f"评估结果: {results}")



# 使用示例  
if __name__ == "__main__":  
    run()