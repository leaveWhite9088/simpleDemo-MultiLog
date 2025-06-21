"""
特征嵌入模块
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的三种特征嵌入

该模块实现论文Section 3.1.2中描述的三路并行特征嵌入：
1. 序列嵌入 (Sequential Embedding) - 事件ID序列的时序模式
2. 量化嵌入 (Quantitative Embedding) - 事件频率统计信息  
3. 语义嵌入 (Semantic Embedding) - 基于FastText的事件语义信息

对应论文Section 3.1.2 - Log Embedding (Three-way Parallel Processing)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
import fasttext
import fasttext.util
from sklearn.feature_extraction.text import TfidfVectorizer


class SequentialEmbedding:
    """
    序列嵌入 (Sequential Embedding)
    
    论文描述：直接使用日志组中的事件ID序列作为输入
    输入：事件ID序列 E_j = (e(s_1), e(s_2), ..., e(s_M))
    输出：事件ID张量，后续通过Embedding层转换为密集向量
    
    对应论文Section 3.1.2.a - Sequential Embedding
    """
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
    
    def embed(self, log_group: List[Dict]) -> torch.Tensor:
        """
        提取事件ID序列
        
        输入：log_group - 日志组S_j
        输出：事件ID序列张量
        
        论文描述：代表了日志发生的时间顺序模式
        """
        # 提取事件ID序列：E_j = (e(s_1), e(s_2), ..., e(s_M))
        event_sequence = [int(log['event_id']) for log in log_group]
        
        event_tensor = torch.tensor(event_sequence, dtype=torch.long)
        
        return event_tensor


class QuantitativeEmbedding:
    """
    量化嵌入 (Quantitative Embedding)
    
    论文描述：统计每个日志组中不同类型日志事件的出现频率
    输入：日志组S_j中的事件ID列表
    输出：计数向量 C_j = (c_j(e_1), c_j(e_2), ..., c_j(e_n))
    
    对应论文Section 3.1.2.b - Quantitative Embedding
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
    
    def embed(self, log_group: List[Dict]) -> torch.Tensor:
        """
        统计事件频率
        
        输入：log_group - 日志组S_j
        输出：事件频率计数向量C_j
        
        论文描述：捕获不同事件类型的出现模式
        """
        # 统计事件ID出现频率
        event_counts = Counter([int(log['event_id']) for log in log_group])
        
        # 构建计数向量 C_j = (c_j(e_1), c_j(e_2), ..., c_j(e_n))
        count_vector = torch.zeros(self.vocab_size)
        for event_id, count in event_counts.items():
            if event_id < self.vocab_size:
                count_vector[event_id] = count
        
        return count_vector


class SemanticEmbedding:
    """
    语义嵌入 (Semantic Embedding)
    
    论文核心创新：提取事件的文本语义信息
    处理流程：
    1. 文本预处理：移除无意义token，停用词过滤，驼峰命名拆分
    2. 词向量化：使用预训练FastText模型(Common Crawl, 300维)
    3. 事件向量化：TF-IDF加权求和聚合单词向量
    
    对应论文Section 3.1.2.c - Semantic Embedding
    输出：语义向量序列 V_j = (v(e(s_1)), v(e(s_2)), ..., v(e(s_M)))
    """
    def __init__(self, embedding_dim: int = 300, model_path: str = None):
        self.embedding_dim = embedding_dim  # FastText 300维
        
        # 论文中提到的停用词列表
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # 加载预训练FastText模型
        # 论文要求：基于Common Crawl Corpus的300维FastText模型
        if model_path:
            self.fasttext_model = fasttext.load_model(model_path)
        else:
            try:
                self.fasttext_model = fasttext.load_model('cc.en.300.bin')
            except:
                print("FastText model not found. Will use random embeddings.")
                self.fasttext_model = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理
        
        论文描述的预处理步骤：
        1. 移除无实际意义的非字符token
        2. 使用驼峰命名法拆分复合词
        3. 过滤停用词
        
        输入：text - 原始日志文本
        输出：预处理后的单词列表
        """
        # 步骤1：移除非字符token
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 步骤2：驼峰命名法拆分
        # 论文描述：使用Camel Case拆分复合词
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        
        # 步骤3：转小写并分词
        words = text.lower().split()
        
        # 步骤4：过滤停用词和短词
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        return words
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        获取单词向量
        
        使用预训练FastText模型获取300维词向量
        如果模型不可用，使用随机向量代替
        
        输入：word - 单词
        输出：300维词向量 v
        """
        if self.fasttext_model:
            try:
                return self.fasttext_model.get_word_vector(word)
            except:
                return np.random.randn(self.embedding_dim) * 0.1
        else:
            return np.random.randn(self.embedding_dim) * 0.1
    
    def embed_single_log(self, log_content: str) -> torch.Tensor:
        """
        单个日志事件的语义嵌入
        
        论文算法：
        1. 文本预处理获取单词列表
        2. 使用FastText获取每个单词的向量v
        3. 使用TF-IDF计算单词权重ε
        4. 加权求和：V = Σ(ε * v)
        
        输入：log_content - 日志文本内容
        输出：事件语义向量V
        """
        # 步骤1：文本预处理
        words = self.preprocess_text(log_content)
        
        if not words:
            return torch.zeros(self.embedding_dim)
        
        # 步骤2：获取词向量
        word_vectors = []
        for word in words:
            word_vectors.append(self.get_word_vector(word))
        
        # 步骤3：TF-IDF权重计算
        # 论文描述：使用TF-IDF计算每个单词在事件中的权重ε
        tfidf = TfidfVectorizer(max_features=len(words))
        try:
            tfidf_matrix = tfidf.fit_transform([' '.join(words)])
            weights = tfidf_matrix.toarray()[0]
        except:
            # 如果TF-IDF失败，使用均匀权重
            weights = np.ones(len(words)) / len(words)
        
        # 步骤4：加权求和
        # 论文公式：V = Σ(ε * v) - 通过加权求和聚合单词向量
        weighted_vectors = []
        for i, vec in enumerate(word_vectors):
            weighted_vectors.append(vec * weights[i])
        
        event_vector = np.mean(weighted_vectors, axis=0)
        
        return torch.tensor(event_vector, dtype=torch.float32)
    
    def embed(self, log_group: List[Dict]) -> torch.Tensor:
        """
        日志组的语义嵌入
        
        输入：log_group - 日志组S_j
        输出：语义向量序列 V_j = (v(e(s_1)), v(e(s_2)), ..., v(e(s_M)))
        
        论文描述：为日志组中每个事件生成语义向量，形成序列
        """
        semantic_vectors = []
        for log in log_group:
            # 为每个日志事件生成语义向量
            vec = self.embed_single_log(log['content'])
            semantic_vectors.append(vec)
        
        # 堆叠成序列张量
        semantic_tensor = torch.stack(semantic_vectors)
        
        return semantic_tensor