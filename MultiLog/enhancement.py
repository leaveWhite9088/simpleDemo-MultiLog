"""
信息增强模块
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的信息增强机制

该模块通过LSTM + Self-Attention机制增强三种维度的特征：
1. 序列特征增强 (Sequential Enhancement)
2. 量化特征增强 (Quantitative Enhancement)  
3. 语义特征增强 (Semantic Enhancement)

对应论文Section 3.1.3 - Information Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SelfAttention(nn.Module):
    """
    Self-Attention机制
    
    论文创新：通过注意力机制捕获日志序列中的长期依赖关系
    注意力打分函数：score(h_m, h_M) = h_m^T * W * h_M
    
    对应论文Section 3.1.3中的注意力机制描述
    """
    def __init__(self, hidden_size: int):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        # 注意力权重矩阵W
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-Attention前向传播
        
        输入：lstm_outputs - LSTM所有时间步的隐藏状态 H = [h_1, h_2, ..., h_M]
        输出：
        - context_vector: 加权上下文向量 c
        - attention_weights: 注意力权重
        
        计算过程：
        1. 计算注意力分数：score(h_m, h_M) = h_m^T * W * h_M
        2. Softmax归一化得到注意力权重
        3. 加权求和得到上下文向量
        """
        batch_size, seq_len, hidden_size = lstm_outputs.size()
        
        # 获取最后一个时间步的隐藏状态h_M作为查询向量
        last_hidden = lstm_outputs[:, -1, :].unsqueeze(1)
        
        # 计算注意力分数：h_m^T * W * h_M
        scores = torch.bmm(lstm_outputs, self.W(last_hidden).transpose(1, 2))
        scores = scores.squeeze(-1)
        
        # Softmax归一化得到注意力权重
        attention_weights = F.softmax(scores, dim=1)
        
        # 加权求和得到上下文向量c
        context_vector = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights


class LSTMWithAttention(nn.Module):
    """
    LSTM + Self-Attention组合模块
    
    论文设计：
    1. LSTM处理序列，捕获时序依赖关系
    2. Self-Attention机制增强重要信息
    3. 拼接上下文向量和最终隐藏状态
    
    对应论文Section 3.1.3 - Information Enhancement
    输出：增强特征 EC = [c; h_M]
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层：处理序列数据
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # 单向LSTM，符合论文描述
        )
        
        # Self-Attention层：增强重要信息
        self.attention = SelfAttention(hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LSTM + Attention前向传播
        
        输入：x - 输入序列
        输出：
        - enhanced_output: 增强特征 EC = [c; h_M]
        - attention_weights: 注意力权重
        
        处理流程：
        1. LSTM处理输入序列
        2. Self-Attention计算上下文向量
        3. 拼接上下文向量和最终隐藏状态
        """
        # LSTM处理：获取所有时间步的隐藏状态
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-Attention：计算加权上下文向量
        context_vector, attention_weights = self.attention(lstm_out)
        
        # 获取最后一个时间步的隐藏状态h_M
        last_hidden = lstm_out[:, -1, :]
        
        # 拼接上下文向量和最终隐藏状态：EC = [c; h_M]
        enhanced_output = torch.cat([context_vector, last_hidden], dim=1)
        
        return enhanced_output, attention_weights


class EnhancementModule(nn.Module):
    """
    信息增强模块
    
    论文核心创新：三路并行的特征增强机制
    1. 序列特征增强：处理事件ID序列的时序模式
    2. 量化特征增强：处理事件频率统计信息
    3. 语义特征增强：处理事件的语义信息
    
    每路都使用LSTM + Self-Attention进行增强
    最终拼接三路输出：[EC_E; EC_C; EC_V]
    
    对应论文Section 3.1.3 - Information Enhancement
    """
    def __init__(self, 
                 event_vocab_size: int = 1000,         # 事件词汇表大小
                 event_embedding_dim: int = 64,        # 事件嵌入维度
                 quantitative_input_size: int = 1000,  # 量化特征维度
                 semantic_input_size: int = 300,       # 语义特征维度(FastText)
                 hidden_size: int = 128,               # LSTM隐藏层大小
                 num_layers: int = 2,                  # LSTM层数
                 dropout: float = 0.2):                # Dropout比例
        super(EnhancementModule, self).__init__()
        
        # 序列特征的事件嵌入层
        # 将事件ID映射为密集向量表示
        self.event_embedding = nn.Embedding(event_vocab_size, event_embedding_dim)
        
        # 第一路：序列特征增强
        # 输入：事件ID序列 E_j，输出：序列增强特征 EC_E
        self.sequential_lstm = LSTMWithAttention(event_embedding_dim, hidden_size, num_layers, dropout)
        
        # 第二路：量化特征增强  
        # 输入：事件频率向量 C_j，输出：量化增强特征 EC_C
        self.quantitative_lstm = LSTMWithAttention(1, hidden_size, num_layers, dropout)
        
        # 第三路：语义特征增强
        # 输入：语义向量序列 V_j，输出：语义增强特征 EC_V
        self.semantic_lstm = LSTMWithAttention(semantic_input_size, hidden_size, num_layers, dropout)
        
        # 输出维度：三路特征拼接后的总维度
        # 每路输出维度 = hidden_size * 2 (上下文向量 + 最终隐藏状态)
        self.output_size = hidden_size * 2 * 3
        
    def forward(self, sequential_input: torch.Tensor, 
                quantitative_input: torch.Tensor, 
                semantic_input: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        三路并行特征增强
        
        输入：
        - sequential_input: 序列特征 E_j
        - quantitative_input: 量化特征 C_j  
        - semantic_input: 语义特征 V_j
        
        输出：
        - combined_output: 拼接的增强特征 [EC_E; EC_C; EC_V]
        - attention_weights: 各路注意力权重
        
        对应论文Section 3.1.3中的三路并行处理流程
        """
        # 第一路：序列特征增强
        # 事件ID序列 -> 事件嵌入 -> LSTM+Attention增强
        sequential_embedded = self.event_embedding(sequential_input)
        sequential_enhanced, seq_attention = self.sequential_lstm(sequential_embedded)
        
        # 第二路：量化特征增强
        # 事件频率向量 -> LSTM+Attention增强
        if len(quantitative_input.shape) == 2:
            quantitative_input = quantitative_input.unsqueeze(-1)
        quant_enhanced, quant_attention = self.quantitative_lstm(quantitative_input)
        
        # 第三路：语义特征增强
        # 语义向量序列 -> LSTM+Attention增强  
        semantic_enhanced, sem_attention = self.semantic_lstm(semantic_input)
        
        # 特征拼接：[EC_E; EC_C; EC_V]
        # 论文描述：将三路增强输出拼接成一个大向量
        combined_output = torch.cat([
            sequential_enhanced,    # EC_E
            quant_enhanced,         # EC_C
            semantic_enhanced       # EC_V
        ], dim=1)
        
        # 收集各路注意力权重用于可解释性分析
        attention_weights = {
            'sequential': seq_attention,
            'quantitative': quant_attention,
            'semantic': sem_attention
        }
        
        return combined_output, attention_weights