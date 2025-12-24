import json  
import numpy as np  
from tqdm import tqdm  
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  
from torch.utils.data import DataLoader  
import evaluate  


from config.evaluator_config import MedQALLMEvaluatorConfig
from config.llm_config import LLMConfig

class MedQALLMEvaluator:  
    def __init__(
        self,
        llm_config: LLMConfig,
        config: MedQALLMEvaluatorConfig):  
        """  
        初始化评估器  
        """  
        self.config = config
        self.llm_config = llm_config
        
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
        
        
        
        # 生成参数配置  
        self.generation_config = {  
            "max_new_tokens": 512,  
            "temperature": 0.7,  
            "top_p": 0.9,  
            "do_sample": True,  
            "pad_token_id": self.tokenizer.eos_token_id  
        }  
        
    
    def format_prompt(self, sample):  
        """构建模型输入格式"""  
        return f"指令：{sample['instruction']}\n问题：{sample['input']}\n回答："  
    
    
    
    def evaluate_dataset(self, dataset_path, batch_size=4, max_samples=100):  
        """  
        评估数据集  
        :param dataset_path: 数据集路径（本地或HuggingFace）  
        :param batch_size: 批量大小  
        :param max_samples: 最大评估样本数（调试用）  
        :return: 评估指标字典  
        """  
        # 加载数据  
        dataset = load_dataset("json", data_files=dataset_path)["train"]  
        dataset = dataset.select(range(min(max_samples, len(dataset))))  
        dataloader = DataLoader(dataset, batch_size=batch_size)  

        results = {  
            "perplexity": [],  
            "bleu": [],  
            "rouge": [],  
            "bertscore": []  
        }  

        with torch.no_grad():  
            for batch in tqdm(dataloader, desc="Evaluating"):  
                # 生成回答  
                prompts = [self.format_prompt(sample) for sample in batch]  
                inputs = self.tokenizer(  
                    prompts,  
                    return_tensors="pt",  
                    padding=True,  
                    truncation=True,  
                    max_length=1024  
                ).to(self.device)  
                
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
                references = batch["output"]  
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
                
                bert_scores = self.bertscore.compute(  
                    predictions=predictions,  
                    references=references,  
                    lang="zh"  
                )  
                results["bertscore"].extend(bert_scores["f1"])  

        # 汇总结果  
        return {  
            "perplexity": np.mean(results["perplexity"]),  
            "bleu": np.mean(results["bleu"]),  
            "rougeL": np.mean(results["rouge"]),  
            "bertscore": np.mean(results["bertscore"])  
        }  

    def save_results(self, results, output_path="eval_results.json"):  
        """保存评估结果"""  
        with open(output_path, "w", encoding="utf-8") as f:  
            json.dump(results, f, ensure_ascii=False, indent=2) 
            
            






# 使用示例  
if __name__ == "__main__":  
    evaluator = MedicalQAEvaluator(  
        model_path=MODEL_PATH,  
        tokenizer_path=TOKENIZER_PATH  
    )  
    
    results = evaluator.evaluate_dataset(  
        dataset_path="path/to/test_dataset.json",  
        batch_size=4,  
        max_samples=100  
    )  
    
    print("评估结果：")  
    for k, v in results.items():  
        print(f"{k}: {v:.4f}")  
    
    evaluator.save_results(results)  
    
    
    # 结果解读建议：  
    # - Perplexity < 30：优秀  
    # - BLEU > 0.25：合格  
    # - BERTScore > 0.75：优秀  