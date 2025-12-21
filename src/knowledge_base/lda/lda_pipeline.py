'''
对一个文档库进行 LDA 主题建模

使用主流框架， 比如 BERTopic 等
'''
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


class LDAPipeline:
    """LDA 主题建模管道"""


    def __init__(self):
        """初始化管道"""
        pass