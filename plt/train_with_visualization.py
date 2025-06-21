import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

from MultiLog.multilog import MultiLog, MultiLogTrainer
from data.dataset import create_dataloader
from plt.visualize_results import MultiLogVisualizer


class VisualizingTrainer:
    def __init__(self, model: MultiLog, trainer: MultiLogTrainer, visualizer: MultiLogVisualizer):
        self.model = model
        self.trainer = trainer
        self.visualizer = visualizer
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.predictions = {
            'y_true': [],
            'y_pred': [],
            'y_scores': []
        }
    
    def train_and_visualize(self, train_loader, val_loader, epochs: int = 20, save_path: str = './checkpoints'):
        """训练模型并生成可视化"""
        print("Starting training with visualization...")
        
        # 先训练AutoEncoder
        print("\nPhase 1: Training AutoEncoder...")
        self.trainer.train_autoencoder(train_loader, epochs=5)
        
        # 训练完整模型
        print("\nPhase 2: Training full MultiLog model...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.trainer.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.trainer.evaluate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, os.path.join(save_path, 'best_model.pth'))
                print(f"  Saved best model with validation accuracy: {val_acc:.4f}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        # 生成预测结果用于可视化
        self._collect_predictions(val_loader)
        
        # 生成所有可视化
        self._generate_visualizations()
    
    def _collect_predictions(self, data_loader):
        """收集模型预测结果"""
        print("\nCollecting predictions for visualization...")
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                if self.model.num_nodes == 1:
                    # 单节点情况
                    for logs, label in batch:
                        # 处理单个节点
                        node_probs = self.model.standalone_estimators[0](logs)
                        # 使用平均概率作为最终分数
                        score = node_probs.mean().item()
                        pred = 1 if score > 0.5 else 0
                        
                        self.predictions['y_true'].append(label)
                        self.predictions['y_pred'].append(pred)
                        self.predictions['y_scores'].append(score)
                else:
                    # 多节点情况
                    for node_logs_list, label in batch:
                        # 获取所有节点的概率
                        all_probs = []
                        for i, logs in enumerate(node_logs_list):
                            if i < len(self.model.standalone_estimators):
                                probs = self.model.standalone_estimators[i](logs)
                                all_probs.append(probs)
                        
                        # 通过集群分类器
                        cluster_output = self.model.cluster_classifier(all_probs)
                        score = torch.sigmoid(cluster_output).item()
                        pred = 1 if score > 0.5 else 0
                        
                        self.predictions['y_true'].append(label)
                        self.predictions['y_pred'].append(pred)
                        self.predictions['y_scores'].append(score)
        
        # 转换为numpy数组
        self.predictions['y_true'] = np.array(self.predictions['y_true'])
        self.predictions['y_pred'] = np.array(self.predictions['y_pred'])
        self.predictions['y_scores'] = np.array(self.predictions['y_scores'])
    
    def _generate_visualizations(self):
        """生成所有可视化图表"""
        print("\nGenerating visualizations...")
        
        # 1. 训练历史
        self.visualizer.plot_training_history(self.history)
        
        # 2. 混淆矩阵
        self.visualizer.plot_confusion_matrix(
            self.predictions['y_true'], 
            self.predictions['y_pred']
        )
        
        # 3. ROC曲线
        self.visualizer.plot_roc_curve(
            self.predictions['y_true'], 
            self.predictions['y_scores']
        )
        
        # 4. PR曲线
        self.visualizer.plot_precision_recall_curve(
            self.predictions['y_true'], 
            self.predictions['y_scores']
        )
        
        # 5. 异常分数分布
        normal_mask = self.predictions['y_true'] == 0
        anomaly_mask = self.predictions['y_true'] == 1
        
        if np.sum(normal_mask) > 0 and np.sum(anomaly_mask) > 0:
            self.visualizer.plot_anomaly_scores_distribution(
                self.predictions['y_scores'][normal_mask],
                self.predictions['y_scores'][anomaly_mask]
            )
        
        print("\nAll visualizations have been saved to ./plt/figures/")


def main():
    parser = argparse.ArgumentParser(description='Train MultiLog with Visualization')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--window_size', type=int, default=5, help='Time window size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save model')
    parser.add_argument('--sample_ratio', type=float, default=0.01, help='Data sampling ratio')
    
    args = parser.parse_args()
    
    print("MultiLog Training with Visualization")
    print(f"Device: {args.device}")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Data sample ratio: {args.sample_ratio*100:.1f}%")
    print("-" * 50)
    
    # 创建数据加载器
    train_loader = create_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        mode='train',
        sample_ratio=args.sample_ratio
    )
    
    val_loader = create_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        mode='val',
        sample_ratio=args.sample_ratio
    )
    
    # 初始化模型
    model = MultiLog(
        num_nodes=args.num_nodes,
        hidden_size=args.hidden_size,
        window_size=args.window_size
    )
    
    # 初始化训练器和可视化器
    trainer = MultiLogTrainer(model, learning_rate=args.learning_rate, device=args.device)
    visualizer = MultiLogVisualizer()
    
    # 创建可视化训练器
    viz_trainer = VisualizingTrainer(model, trainer, visualizer)
    
    # 开始训练和可视化
    viz_trainer.train_and_visualize(
        train_loader, 
        val_loader, 
        epochs=args.epochs,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()