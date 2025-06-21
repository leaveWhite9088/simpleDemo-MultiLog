"""
独立估计模块
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的第一阶段：Standalone Estimation

该模块在每个数据库节点上独立运行，负责：
1. 日志解析与分组
2. 三种维度的特征嵌入提取（序列、量化、语义）
3. 通过增强模型生成异常概率列表

对应论文Section 3.1 - Standalone Estimation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from .log_parser import LogParser, LogGrouper
from .embeddings import SequentialEmbedding, QuantitativeEmbedding, SemanticEmbedding
from .enhancement import EnhancementModule


class StandaloneEstimation(nn.Module):
    """
    独立估计模块 - MultiLog第一阶段
    
    论文核心创新：多维度特征提取和信息增强
    处理流程：
    1. 日志解析与分组（Drain3 + 固定时间窗口）
    2. 三路并行特征嵌入提取
    3. LSTM + Self-Attention增强
    4. 异常概率预测
    
    对应论文Section 3.1 - Standalone Estimation
    输出：概率列表 P_i = [p_1, p_2, ..., p_k_i]
    """
    def __init__(self,
                 event_vocab_size: int = 1000,      # 事件词汇表大小
                 event_embedding_dim: int = 64,     # 事件嵌入维度
                 semantic_dim: int = 300,           # 语义嵌入维度(FastText 300维)
                 hidden_size: int = 128,            # LSTM隐藏层大小
                 num_layers: int = 2,               # LSTM层数
                 dropout: float = 0.2,              # Dropout比例
                 window_size: int = 5):             # 时间窗口大小(秒)
        super(StandaloneEstimation, self).__init__()
        
        # 步骤1：日志解析与分组
        # 对应论文Section 3.1.1 - Log Parsing and Grouping
        self.log_parser = LogParser()              # Drain3日志解析器
        self.log_grouper = LogGrouper(window_size=window_size)  # 固定时间窗口分组
        
        # 步骤2：三路并行特征嵌入
        # 对应论文Section 3.1.2 - Log Embedding
        self.sequential_embedding = SequentialEmbedding(max_events=event_vocab_size)
        self.quantitative_embedding = QuantitativeEmbedding(vocab_size=event_vocab_size)
        self.semantic_embedding = SemanticEmbedding(embedding_dim=semantic_dim)
        
        # 步骤3：信息增强模块
        # 对应论文Section 3.1.3 - Information Enhancement
        self.enhancement_module = EnhancementModule(
            event_vocab_size=event_vocab_size,
            event_embedding_dim=event_embedding_dim,
            quantitative_input_size=event_vocab_size,
            semantic_input_size=semantic_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 步骤4：异常概率预测器
        # 对应论文Section 3.1.4 - Anomaly Probability Prediction
        self.classifier = nn.Sequential(
            nn.Linear(self.enhancement_module.output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出[0,1]范围的异常概率
        )
        
    def process_raw_logs(self, log_lines: List[str]) -> List[List[Dict]]:
        """
        原始日志预处理
        
        步骤：
        1. 使用Drain3解析非结构化日志为结构化事件
        2. 采用固定时间窗口分组
        
        输入：log_lines - 原始日志文本列表
        输出：分组后的日志事件列表
        
        对应论文Section 3.1.1 - Log Parsing and Grouping
        """
        # Drain3日志解析：将非结构化日志解析为结构化事件
        parsed_logs = self.log_parser.parse_logs(log_lines)
        
        # 固定时间窗口分组：将事件序列切分为固定长度的日志组S_j
        log_groups = self.log_grouper.group_logs_by_window(parsed_logs)
        
        return log_groups
    
    def forward_single_group(self, log_group: List[Dict]) -> Tuple[torch.Tensor, dict]:
        """
        单个日志组的前向传播
        
        处理流程：
        1. 三路并行特征嵌入提取
        2. LSTM + Self-Attention增强
        3. 异常概率预测
        
        输入：log_group - 单个日志组S_j
        输出：
        - anomaly_prob: 异常概率p_j
        - attention_weights: 注意力权重
        
        对应论文Section 3.1.2-3.1.4的完整处理流程
        """
        # 步骤1：三路并行特征嵌入
        # 对应论文Section 3.1.2 - Log Embedding
        
        # 序列嵌入：事件ID序列 E_j = (e(s_1), e(s_2), ..., e(s_M))
        sequential_emb = self.sequential_embedding.embed(log_group)
        
        # 量化嵌入：事件频率计数向量 C_j = (c_j(e_1), c_j(e_2), ..., c_j(e_n))
        quantitative_emb = self.quantitative_embedding.embed(log_group)
        
        # 语义嵌入：语义向量序列 V_j = (v(e(s_1)), v(e(s_2)), ..., v(e(s_M)))
        semantic_emb = self.semantic_embedding.embed(log_group)
        
        # 维度调整以适配LSTM输入要求
        if len(sequential_emb.shape) == 1:
            sequential_emb = sequential_emb.unsqueeze(0)
        if len(quantitative_emb.shape) == 1:
            quantitative_emb = quantitative_emb.unsqueeze(0).unsqueeze(1)
        if len(semantic_emb.shape) == 2:
            semantic_emb = semantic_emb.unsqueeze(0)
        
        # 步骤2：信息增强 (LSTM + Self-Attention)
        # 对应论文Section 3.1.3 - Information Enhancement
        enhanced_features, attention_weights = self.enhancement_module(
            sequential_emb, quantitative_emb, semantic_emb
        )
        
        # 步骤3：异常概率预测
        # 对应论文Section 3.1.4 - Anomaly Probability Prediction
        anomaly_prob = self.classifier(enhanced_features)
        
        return anomaly_prob.squeeze(), attention_weights
    
    def forward(self, log_groups: List[List[Dict]]) -> List[float]:
        """
        多个日志组的批量处理
        
        输入：log_groups - 多个日志组列表
        输出：异常概率列表 P_i = [p_1, p_2, ..., p_k_i]
        
        对应论文中单个节点的完整处理流程
        """
        probabilities = []
        
        for group in log_groups:
            if len(group) > 0:
                # 处理单个日志组
                prob, _ = self.forward_single_group(group)
                probabilities.append(prob.item() if isinstance(prob, torch.Tensor) else prob)
            else:
                # 空日志组赋予0概率
                probabilities.append(0.0)
        
        return probabilities
    
    def process_node_logs(self, log_lines: List[str]) -> List[float]:
        """
        处理单个节点的完整日志
        
        完整流程：
        1. 原始日志预处理（解析+分组）
        2. 特征嵌入提取
        3. 信息增强
        4. 异常概率预测
        
        输入：log_lines - 节点的原始日志文本
        输出：异常概率列表 P_i
        
        对应论文Section 3.1中描述的完整Standalone Estimation流程
        """
        # 日志预处理
        log_groups = self.process_raw_logs(log_lines)
        
        # 特征提取和异常检测
        return self.forward(log_groups)