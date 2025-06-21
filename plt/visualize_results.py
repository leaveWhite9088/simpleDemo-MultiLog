import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from typing import List, Dict, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


class MultiLogVisualizer:
    def __init__(self, save_dir: str = './plt/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格 - 兼容不同版本的matplotlib/seaborn
        try:
            # 尝试使用seaborn样式
            available_styles = plt.style.available
            if 'seaborn-v0_8-darkgrid' in available_styles:
                plt.style.use('seaborn-v0_8-darkgrid')
            elif 'seaborn-darkgrid' in available_styles:
                plt.style.use('seaborn-darkgrid')
            elif 'seaborn' in available_styles:
                plt.style.use('seaborn')
            else:
                # 使用默认样式
                plt.style.use('default')
                plt.rcParams['axes.grid'] = True
                plt.rcParams['grid.alpha'] = 0.3
            
            # 设置seaborn调色板
            sns.set_palette("husl")
        except Exception as e:
            print(f"Warning: Could not set plot style: {e}")
            # 使用基本的matplotlib设置
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    def plot_training_history(self, history: Dict[str, List[float]], save_name: str = 'training_history.png'):
        """绘制训练历史曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss During Training', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy During Training', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              save_name: str = 'confusion_matrix.png'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # 添加准确率等指标
        accuracy = np.trace(cm) / np.sum(cm)
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        plt.text(2.5, 0.5, metrics_text, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                       save_name: str = 'roc_curve.png'):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=1)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   save_name: str = 'pr_curve.png'):
        """绘制Precision-Recall曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 计算平均精度
        avg_precision = np.mean(precision)
        plt.text(0.5, 0.1, f'Average Precision: {avg_precision:.3f}', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curve saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_anomaly_scores_distribution(self, normal_scores: np.ndarray, anomaly_scores: np.ndarray,
                                       save_name: str = 'score_distribution.png'):
        """绘制异常分数分布"""
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Anomaly Scores', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Normal: μ={np.mean(normal_scores):.3f}, σ={np.std(normal_scores):.3f}\n'
        stats_text += f'Anomaly: μ={np.mean(anomaly_scores):.3f}, σ={np.std(anomaly_scores):.3f}'
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score distribution plot saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                               save_name: str = 'feature_importance.png'):
        """绘制特征重要性"""
        # 排序特征
        indices = np.argsort(importance_scores)[::-1][:20]  # 只显示前20个最重要的特征
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importance_scores[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 20 Feature Importance', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {os.path.join(self.save_dir, save_name)}")
    
    def plot_time_series_anomalies(self, timestamps: List[str], values: np.ndarray, 
                                  anomaly_indices: List[int], save_name: str = 'time_series_anomalies.png'):
        """绘制时间序列中的异常"""
        plt.figure(figsize=(15, 6))
        
        # 绘制正常数据点
        normal_mask = np.ones(len(values), dtype=bool)
        normal_mask[anomaly_indices] = False
        plt.plot(timestamps[normal_mask], values[normal_mask], 'b-', label='Normal', linewidth=1)
        
        # 高亮异常点
        if len(anomaly_indices) > 0:
            plt.scatter(np.array(timestamps)[anomaly_indices], values[anomaly_indices], 
                       color='red', s=100, label='Anomaly', zorder=5)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Time Series with Detected Anomalies', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Time series anomalies plot saved to {os.path.join(self.save_dir, save_name)}")


def load_training_history(checkpoint_path: str) -> Dict[str, List[float]]:
    """从检查点文件加载训练历史"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'history' in checkpoint:
            return checkpoint['history']
    return {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }


def main():
    """主函数：演示可视化功能"""
    visualizer = MultiLogVisualizer()
    
    # 生成示例数据进行可视化
    print("Generating example visualizations...")
    
    # 1. 训练历史
    history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2],
        'val_loss': [0.85, 0.65, 0.55, 0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43],
        'train_acc': [0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.88, 0.89, 0.9],
        'val_acc': [0.55, 0.65, 0.7, 0.73, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8]
    }
    visualizer.plot_training_history(history)
    
    # 2. 混淆矩阵
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 10)
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 10)
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # 3. ROC曲线
    y_scores = np.random.rand(len(y_true))
    y_scores[y_true == 1] += 0.3  # 让异常样本的分数更高
    visualizer.plot_roc_curve(y_true, y_scores)
    
    # 4. PR曲线
    visualizer.plot_precision_recall_curve(y_true, y_scores)
    
    # 5. 异常分数分布
    normal_scores = np.random.normal(0.3, 0.1, 1000)
    anomaly_scores = np.random.normal(0.7, 0.15, 300)
    visualizer.plot_anomaly_scores_distribution(normal_scores, anomaly_scores)
    
    print("\nAll visualizations have been generated successfully!")


if __name__ == '__main__':
    main()