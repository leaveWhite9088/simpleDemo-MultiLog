"""
集群分类器模块
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的第二阶段：Cluster Classifier

该模块包含两个核心组件：
1. 概率标准化AutoEncoder - 解决不同长度概率列表的标准化问题
2. 元分类器 - 基于标准化特征对整个集群状态进行分类

对应论文Section 3.2 - Cluster Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ProbabilityAutoEncoder(nn.Module):
    """
    概率标准化AutoEncoder
    
    论文核心创新：解决各节点输出概率列表长度不一致的问题
    - 编码器：将变长概率列表映射到固定长度的隐向量
    - 解码器：从隐向量重建原始概率列表
    
    对应论文Section 3.2.1 - Probability Standardization
    参数设置：
    - β (input_size): 128 - 标准化后的固定长度
    - μ (hidden_size): 32 - 隐向量维度
    """
    def __init__(self, input_size: int = 128, hidden_size: int = 32):
        super(ProbabilityAutoEncoder, self).__init__()
        self.input_size = input_size      # β: 论文中的标准化长度
        self.hidden_size = hidden_size    # μ: 论文中的隐向量大小
        
        # 编码器：f_enc，将β维输入映射到μ维隐空间
        # 论文描述：三个带ReLU激活函数的线性层
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, hidden_size)
        )
        
        # 解码器：f_dec，从μ维隐空间重建β维输出
        # 用于训练时的重建损失计算
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()  # 确保输出在[0,1]范围内，符合概率特性
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码过程：P_i -> Z_i"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码过程：Z_i -> P_i'"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AutoEncoder前向传播
        
        输入：x - 填充后的概率列表 (batch_size, β)
        输出：
        - x_reconstructed: 重建的概率列表
        - z: 编码后的隐向量 (batch_size, μ)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def standardize_probabilities(self, prob_lists: List[List[float]]) -> torch.Tensor:
        """
        概率列表标准化
        
        解决论文中提到的核心问题：各节点概率列表长度不一致
        方法：填充(padding)或截断(truncation)到固定长度β
        
        输入：prob_lists - 各节点的概率列表 {P_1, P_2, ..., P_N}
        输出：标准化的隐向量 {Z_1, Z_2, ..., Z_N}
        """
        padded_probs = []
        for probs in prob_lists:
            if len(probs) < self.input_size:
                # 填充零值到固定长度β
                padded = probs + [0.0] * (self.input_size - len(probs))
            else:
                # 截断到固定长度β
                padded = probs[:self.input_size]
            padded_probs.append(padded)
        
        prob_tensor = torch.tensor(padded_probs, dtype=torch.float32)
        
        # 编码为固定长度的隐向量
        with torch.no_grad():
            _, z = self.forward(prob_tensor)
        
        return z


class MetaClassifier(nn.Module):
    """
    元分类器
    
    基于标准化的隐向量对整个集群状态进行分类
    输入：拼接后的所有节点隐向量 Z = [Z_1; Z_2; ...; Z_N]
    输出：集群状态预测 - "Normal" 或 "Anomalous"
    
    对应论文Section 3.2.2 - Meta-Classification
    """
    def __init__(self, num_nodes: int, hidden_size: int = 32, dropout: float = 0.2):
        super(MetaClassifier, self).__init__()
        
        # 输入维度：N个节点 × μ维隐向量
        input_size = num_nodes * hidden_size
        
        # 元分类器网络：f_meta
        # 论文描述：包含一个隐藏层和Softmax输出层的神经网络
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 二分类：Normal(0) / Anomalous(1)
        )
    
    def forward(self, concatenated_features: torch.Tensor) -> torch.Tensor:
        """
        元分类器前向传播
        
        输入：concatenated_features - 拼接的隐向量 Z
        输出：分类概率分布 [P(Normal), P(Anomalous)]
        """
        logits = self.classifier(concatenated_features)
        return F.softmax(logits, dim=-1)


class ClusterClassifier(nn.Module):
    """
    集群分类器 - MultiLog第二阶段
    
    整合概率标准化和元分类功能
    实现论文Figure 6中第二阶段的完整流程：
    1. 概率列表标准化 (AutoEncoder)
    2. 特征拼接
    3. 元分类 (Meta-Classification)
    
    对应论文Section 3.2 - Cluster Classifier
    """
    def __init__(self, 
                 num_nodes: int,
                 prob_list_size: int = 128,    # β: AutoEncoder输入长度
                 hidden_size: int = 32,        # μ: AutoEncoder隐向量大小
                 dropout: float = 0.2):
        super(ClusterClassifier, self).__init__()
        
        self.num_nodes = num_nodes
        
        # 组件1：概率标准化AutoEncoder
        self.autoencoder = ProbabilityAutoEncoder(prob_list_size, hidden_size)
        
        # 组件2：元分类器
        self.meta_classifier = MetaClassifier(num_nodes, hidden_size, dropout)
    
    def forward(self, node_probabilities: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        集群分类器前向传播
        
        输入：node_probabilities - 各节点概率列表 {P_1, P_2, ..., P_N}
        输出：
        - cluster_prediction: 集群状态预测
        - standardized_features: 标准化后的特征
        
        对应论文Algorithm 1中第二阶段的处理流程
        """
        # 步骤1：概率标准化
        # 将变长概率列表标准化为固定长度的隐向量
        standardized_features = self.autoencoder.standardize_probabilities(node_probabilities)
        
        # 步骤2：特征拼接
        # 将所有节点的隐向量拼接成单一向量 Z = [Z_1; Z_2; ...; Z_N]
        concatenated = standardized_features.flatten()
        if len(concatenated.shape) == 1:
            concatenated = concatenated.unsqueeze(0)
        
        # 步骤3：元分类
        # 基于拼接特征预测集群状态
        cluster_prediction = self.meta_classifier(concatenated)
        
        return cluster_prediction, standardized_features
    
    def predict(self, node_probabilities: List[List[float]]) -> str:
        """
        预测接口
        
        输入：node_probabilities - 各节点概率列表
        输出：字符串形式的预测结果 "Normal" 或 "Anomalous"
        
        对应论文中的最终集群状态判断
        """
        with torch.no_grad():
            predictions, _ = self.forward(node_probabilities)
            _, predicted_class = torch.max(predictions, dim=-1)
            
            # 0: Normal, 1: Anomalous
            return "Normal" if predicted_class.item() == 0 else "Anomalous"