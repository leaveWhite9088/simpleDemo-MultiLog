import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Dict, Tuple
import random
import numpy as np
import re


class Single2SingleDataset(Dataset):
    def __init__(self, data_path: str, mode: str = 'train', train_ratio: float = 0.8):
        self.data_path = data_path
        self.mode = mode
        self.train_ratio = train_ratio
        
        # 异常类型映射
        self.anomaly_types = {
            'cpu_continue': 1, 'cpu_continue_leader': 1,
            'io_continue': 2, 'io_continue_leader': 2,
            'memory_continue': 3, 'memory_continue_leader': 3,
            'network2_continue': 4, 'network2_continue_leader': 4,
            'network3_continue': 5, 'network3_continue_leader': 5,
            'shutdown_continue': 6, 'shutdown_continue_leader': 6,
            'query_continue': 7, 'query_continue_leader': 7,
            'export_continue': 8, 'export_continue_leader': 8,
            'import_continue': 9, 'import_continue_leader': 9,
            'compaction_continue': 10, 'compaction_continue_leader': 10,
            'flush_continue': 11, 'flush_continue_leader': 11
        }
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """加载数据，支持多种数据格式"""
        single2single_path = os.path.join(self.data_path, 'Single2Single')
        
        # 优先从Single2Single目录加载实际数据集
        if os.path.exists(single2single_path):
            all_files = self._load_single2single_data(single2single_path)
        # 备选方案：从normal/anomaly目录加载
        elif os.path.exists(os.path.join(self.data_path, 'normal')):
            normal_files = self._load_files_from_dir(os.path.join(self.data_path, 'normal'), label=0)
            anomaly_files = self._load_files_from_dir(os.path.join(self.data_path, 'anomaly'), label=1)
            all_files = normal_files + anomaly_files
        # 从JSON文件加载
        elif os.path.exists(os.path.join(self.data_path, 'data.json')):
            with open(os.path.join(self.data_path, 'data.json'), 'r') as f:
                data = json.load(f)
                all_files = [(item['logs'], item['label']) for item in data]
        # 生成合成数据
        else:
            print("未找到实际数据集，生成合成数据用于演示...")
            all_files = self._generate_synthetic_data()
        
        if not all_files:
            print("警告：未找到任何数据文件，生成合成数据...")
            all_files = self._generate_synthetic_data()
        
        random.shuffle(all_files)
        
        # 分割训练和验证数据
        split_idx = int(len(all_files) * self.train_ratio)
        if self.mode == 'train':
            selected_files = all_files[:split_idx]
        else:
            selected_files = all_files[split_idx:]
        
        for logs, label in selected_files:
            self.samples.append(logs)
            self.labels.append(label)
        
        print(f"加载{self.mode}数据: {len(self.samples)}个样本")
    
    def _load_single2single_data(self, single2single_path: str) -> List[Tuple[List[str], int]]:
        """加载Single2Single数据集"""
        all_files = []
        
        # 遍历所有异常类型目录
        for item_name in os.listdir(single2single_path):
            item_path = os.path.join(single2single_path, item_name)
            
            # 跳过非目录文件
            if not os.path.isdir(item_path):
                continue
            
            # 确定标签
            if item_name in self.anomaly_types:
                label = 1  # 异常
                print(f"加载异常数据: {item_name}")
            else:
                label = 0  # 正常（如果有的话）
                print(f"加载正常数据: {item_name}")
            
            # 加载目录中的所有日志文件
            files = self._load_files_from_dir(item_path, label)
            all_files.extend(files)
        
        # 如果没有多文件，尝试直接读取单个文件
        for item_name in os.listdir(single2single_path):
            item_path = os.path.join(single2single_path, item_name)
            
            if os.path.isfile(item_path) and not item_name.endswith('.tar.gz'):
                # 根据文件名确定标签
                if any(anomaly_type in item_name for anomaly_type in self.anomaly_types.keys()):
                    label = 1
                else:
                    label = 0
                
                try:
                    with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                        logs = f.readlines()
                    if logs:  # 确保文件不为空
                        all_files.append((logs, label))
                        print(f"加载单个文件: {item_name} (标签: {label})")
                except Exception as e:
                    print(f"读取文件 {item_name} 时出错: {e}")
        
        return all_files
    
    def _load_files_from_dir(self, dir_path: str, label: int) -> List[Tuple[List[str], int]]:
        """从目录加载日志文件"""
        files = []
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                # 跳过子目录
                if os.path.isdir(file_path):
                    continue
                
                # 只处理日志文件
                if filename.endswith(('.log', '.txt')) or not '.' in filename:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            logs = f.readlines()
                        if logs:  # 确保文件不为空
                            files.append((logs, label))
                    except Exception as e:
                        print(f"读取文件 {filename} 时出错: {e}")
        return files
    
    def _generate_synthetic_data(self) -> List[Tuple[List[str], int]]:
        """生成合成数据用于演示"""
        synthetic_data = []
        
        normal_templates = [
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} INFO Database connection established",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} INFO Query executed successfully",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} INFO Transaction committed",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} INFO Cache updated",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} INFO Health check passed"
        ]
        
        anomaly_templates = [
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} ERROR Database connection failed",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} ERROR Query timeout exceeded",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} ERROR Transaction rollback",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} WARN High memory usage detected",
            "2024-01-{:02d} {:02d}:{:02d}:{:02d} ERROR Deadlock detected"
        ]
        
        for i in range(100):
            if i < 70:  # 70%正常数据
                logs = []
                for j in range(50):
                    template = random.choice(normal_templates)
                    log = template.format(
                        random.randint(1, 28),
                        random.randint(0, 23),
                        random.randint(0, 59),
                        random.randint(0, 59)
                    )
                    logs.append(log + "\n")
                synthetic_data.append((logs, 0))
            else:  # 30%异常数据
                logs = []
                for j in range(50):
                    if random.random() < 0.3:  # 30%概率生成异常日志
                        template = random.choice(anomaly_templates)
                    else:
                        template = random.choice(normal_templates)
                    log = template.format(
                        random.randint(1, 28),
                        random.randint(0, 23),
                        random.randint(0, 59),
                        random.randint(0, 59)
                    )
                    logs.append(log + "\n")
                synthetic_data.append((logs, 1))
        
        return synthetic_data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class MultiNodeDataset(Dataset):
    def __init__(self, data_path: str, num_nodes: int = 1, mode: str = 'train', train_ratio: float = 0.8):
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.mode = mode
        self.train_ratio = train_ratio
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """为多节点配置加载数据"""
        base_dataset = Single2SingleDataset(self.data_path, self.mode, self.train_ratio)
        
        for i in range(len(base_dataset)):
            node_logs = []
            for _ in range(self.num_nodes):
                idx = random.randint(0, len(base_dataset) - 1)
                logs, _ = base_dataset[idx]
                node_logs.append(logs)
            
            _, label = base_dataset[i]
            self.samples.append(node_logs)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def create_dataloader(data_path: str, 
                     batch_size: int = 1,
                     num_nodes: int = 1,
                     mode: str = 'train',
                     train_ratio: float = 0.8) -> DataLoader:
    """创建数据加载器"""
    
    if num_nodes == 1:
        dataset = Single2SingleDataset(data_path, mode, train_ratio)
    else:
        dataset = MultiNodeDataset(data_path, num_nodes, mode, train_ratio)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=0,
        collate_fn=lambda x: x
    )