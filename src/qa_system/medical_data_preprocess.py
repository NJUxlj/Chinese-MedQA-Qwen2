from langchain.text_splitter import ChineseRecursiveTextSplitter  

medical_splitter = ChineseRecursiveTextSplitter(  
    chunk_size=512,  
    chunk_overlap=64,  
    separators=["\n\n", "。", "！", "？", "；"], # 保留医学段落结构  
    keep_separator=True  
)  

def process_medical_doc(text):  
    # 过滤非医学内容  
    if "患者" not in text and "治疗" not in text:  
        return []  
    
    # 特殊处理表格数据  
    if "<table>" in text:  
        return process_medical_tables(text)  
    
    return medical_splitter.split_text(text)  