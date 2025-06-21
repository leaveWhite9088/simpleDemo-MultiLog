"""
MultiLog模型包
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》

模块结构：
├── multilog.py - 主模型文件，实现两阶段框架
├── standalone_estimation.py - 第一阶段：独立估计模块
├── cluster_classifier.py - 第二阶段：集群分类器
├── enhancement.py - 信息增强模块（LSTM + Self-Attention）
├── embeddings.py - 三种特征嵌入（序列、量化、语义）
└── log_parser.py - 日志解析与分组模块

对应论文架构：
- Section 3.1: Standalone Estimation (第一阶段)
- Section 3.2: Cluster Classifier (第二阶段)
"""

from .multilog import MultiLog, MultiLogTrainer
from .standalone_estimation import StandaloneEstimation
from .cluster_classifier import ClusterClassifier, ProbabilityAutoEncoder, MetaClassifier
from .enhancement import EnhancementModule, LSTMWithAttention, SelfAttention
from .embeddings import SequentialEmbedding, QuantitativeEmbedding, SemanticEmbedding
from .log_parser import LogParser, LogGrouper

__all__ = [
    # 主模型
    'MultiLog',
    'MultiLogTrainer',
    
    # 第一阶段模块
    'StandaloneEstimation',
    'EnhancementModule',
    'LSTMWithAttention', 
    'SelfAttention',
    
    # 特征嵌入
    'SequentialEmbedding',
    'QuantitativeEmbedding', 
    'SemanticEmbedding',
    
    # 第二阶段模块
    'ClusterClassifier',
    'ProbabilityAutoEncoder',
    'MetaClassifier',
    
    # 日志预处理
    'LogParser',
    'LogGrouper'
]
