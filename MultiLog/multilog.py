"""
MultiLog模型主文件
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的MultiLog框架

该文件实现了论文中描述的两阶段监督学习框架：
1. 第一阶段：独立评估 (Standalone Estimation) - 每个数据库节点独立运行
2. 第二阶段：集群分类器 (Cluster Classifier) - 聚合所有节点信息进行集群状态分类

论文对应架构：Figure 6 - MultiLog框架总览图
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from .standalone_estimation import StandaloneEstimation
from .cluster_classifier import ClusterClassifier


class MultiLog(nn.Module):
    """
    MultiLog多元日志异常检测模型
    
    论文核心创新：针对分布式数据库环境设计的两阶段异常检测框架
    - 第一阶段：每个节点独立提取序列、量化、语义三种特征
    - 第二阶段：通过AutoEncoder标准化概率列表，元分类器做最终判决
    
    对应论文章节：Section 3 - The MultiLog Framework
    """
    def __init__(self,
                 num_nodes: int = 1,                    # 分布式数据库节点数量
                 event_vocab_size: int = 1000,          # 日志事件词汇表大小
                 event_embedding_dim: int = 64,         # 事件嵌入维度
                 semantic_dim: int = 300,               # 语义嵌入维度(FastText 300维)
                 hidden_size: int = 128,                # LSTM隐藏层大小
                 num_layers: int = 2,                   # LSTM层数
                 dropout: float = 0.2,                  # Dropout比例
                 window_size: int = 5,                  # 时间窗口大小(秒)
                 prob_list_size: int = 128,             # AutoEncoder输入长度β
                 autoencoder_hidden_size: int = 32):    # AutoEncoder隐向量大小μ
        super(MultiLog, self).__init__()
        
        self.num_nodes = num_nodes
        
        # 第一阶段：为每个节点创建独立估计器
        # 对应论文Section 3.1 - Standalone Estimation
        self.standalone_estimators = nn.ModuleList([
            StandaloneEstimation(
                event_vocab_size=event_vocab_size,
                event_embedding_dim=event_embedding_dim,
                semantic_dim=semantic_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                window_size=window_size
            ) for _ in range(num_nodes)
        ])
        
        # 第二阶段：集群分类器
        # 对应论文Section 3.2 - Cluster Classifier
        self.cluster_classifier = ClusterClassifier(
            num_nodes=num_nodes,
            prob_list_size=prob_list_size,
            hidden_size=autoencoder_hidden_size,
            dropout=dropout
        )
    
    def forward(self, node_logs: List[List[str]]) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        MultiLog前向传播
        
        输入：node_logs - 每个节点的原始日志列表
        输出：
        - cluster_prediction: 集群异常预测结果 (Normal/Anomalous)
        - all_node_probabilities: 每个节点的概率列表
        
        对应论文Algorithm 1 - MultiLog Training Process
        """
        if len(node_logs) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {len(node_logs)}")
        
        # 第一阶段：独立评估
        # 每个节点独立处理日志，提取序列、量化、语义特征
        # 输出每个节点的异常概率列表 P_i = [p_1, p_2, ..., p_k_i]
        all_node_probabilities = []
        
        for i, (logs, estimator) in enumerate(zip(node_logs, self.standalone_estimators)):
            # 对应论文Section 3.1中的特征提取和增强过程
            node_probs = estimator.process_node_logs(logs)
            all_node_probabilities.append(node_probs)
        
        # 第二阶段：集群分类
        # 通过AutoEncoder标准化概率列表，元分类器做最终判决
        # 对应论文Section 3.2中的概率标准化和元分类过程
        cluster_prediction, standardized_features = self.cluster_classifier(all_node_probabilities)
        
        return cluster_prediction, all_node_probabilities
    
    def predict(self, node_logs: List[List[str]]) -> str:
        """
        预测接口，返回字符串结果
        
        输出："Normal" 或 "Anomalous"
        对应论文中的最终集群状态判断
        """
        with torch.no_grad():
            cluster_prediction, _ = self.forward(node_logs)
            return self.cluster_classifier.predict([[] for _ in range(self.num_nodes)])


class MultiLogTrainer:
    """
    MultiLog训练器
    
    实现论文中的两阶段训练策略：
    1. 先训练AutoEncoder学习概率列表的紧凑表示
    2. 再端到端训练整个模型进行异常分类
    
    对应论文Section 4.3 - Training Strategy
    """
    def __init__(self, model: MultiLog, learning_rate: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # AutoEncoder单独训练的优化器
        # 对应论文中提到的"预训练AutoEncoder以获得稳定的特征表示"
        self.ae_optimizer = torch.optim.Adam(
            model.cluster_classifier.autoencoder.parameters(), 
            lr=learning_rate
        )
        self.ae_criterion = nn.MSELoss()
    
    def train_autoencoder(self, dataloader, epochs: int = 10):
        """
        第一阶段训练：AutoEncoder预训练
        
        目标：学习将不同长度的概率列表映射到固定长度的隐向量
        损失函数：均方误差损失 (MSE Loss)
        
        对应论文Section 3.2.1 - Probability Standardization
        """
        print("Training AutoEncoder...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                all_probabilities = []
                
                # 为每个节点生成概率列表
                for node_logs, _ in batch:
                    if isinstance(node_logs[0], list):
                        for i, logs in enumerate(node_logs):
                            probs = self.model.standalone_estimators[i].process_node_logs(logs)
                            all_probabilities.append(probs)
                    else:
                        probs = self.model.standalone_estimators[0].process_node_logs(node_logs)
                        all_probabilities.append(probs)
                
                if all_probabilities:
                    # 概率列表标准化
                    prob_tensor = self.model.cluster_classifier.autoencoder.standardize_probabilities([all_probabilities[0]])
                    
                    # 填充到固定长度β
                    padded_input = torch.zeros(1, self.model.cluster_classifier.autoencoder.input_size).to(self.device)
                    for i, p in enumerate(all_probabilities[0][:self.model.cluster_classifier.autoencoder.input_size]):
                        padded_input[0, i] = p
                    
                    # AutoEncoder重建
                    reconstructed, _ = self.model.cluster_classifier.autoencoder(padded_input)
                    loss = self.ae_criterion(reconstructed, padded_input)
                    
                    self.ae_optimizer.zero_grad()
                    loss.backward()
                    self.ae_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"AutoEncoder Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def train_epoch(self, dataloader):
        """
        第二阶段训练：端到端训练整个MultiLog模型
        
        目标：优化集群异常检测性能
        损失函数：交叉熵损失 (Cross-Entropy Loss)
        
        对应论文Section 4.3中的端到端训练策略
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            batch_loss = 0
            batch_correct = 0
            batch_total = 0
            
            for sample in batch:
                node_logs, label = sample
                
                # 单节点情况处理
                if self.model.num_nodes == 1:
                    node_logs = [node_logs]
                
                try:
                    # MultiLog前向传播
                    cluster_prediction, _ = self.model(node_logs)
                    
                    label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)
                    
                    # 计算分类损失
                    loss = self.criterion(cluster_prediction.unsqueeze(0), label_tensor)
                    
                    batch_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(cluster_prediction, 0)
                    batch_correct += (predicted == label_tensor).sum().item()
                    batch_total += 1
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
            
            # 批次更新
            if batch_total > 0:
                avg_batch_loss = batch_loss / batch_total
                self.optimizer.zero_grad()
                avg_batch_loss.backward()
                self.optimizer.step()
                
                total_loss += avg_batch_loss.item()
                correct += batch_correct
                total += batch_total
        
        return total_loss / max(total, 1), correct / max(total, 1)
    
    def evaluate(self, dataloader):
        """
        模型评估
        
        计算测试集上的损失和准确率
        对应论文Section 5中的实验评估方法
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                for sample in batch:
                    node_logs, label = sample
                    
                    if self.model.num_nodes == 1:
                        node_logs = [node_logs]
                    
                    try:
                        cluster_prediction, _ = self.model(node_logs)
                        
                        label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)
                        
                        loss = self.criterion(cluster_prediction.unsqueeze(0), label_tensor)
                        total_loss += loss.item()
                        
                        _, predicted = torch.max(cluster_prediction, 0)
                        correct += (predicted == label_tensor).sum().item()
                        total += 1
                        
                    except Exception as e:
                        print(f"Error evaluating sample: {e}")
                        continue
        
        return total_loss / max(total, 1), correct / max(total, 1)