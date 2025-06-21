import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Dict, Tuple
import random
import numpy as np


class Single2SingleDataset(Dataset):
    def __init__(self, data_path: str, mode: str = 'train', train_ratio: float = 0.8):
        self.data_path = data_path
        self.mode = mode
        self.train_ratio = train_ratio
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        if os.path.exists(os.path.join(self.data_path, 'normal')):
            normal_files = self._load_files_from_dir(os.path.join(self.data_path, 'normal'), label=0)
            anomaly_files = self._load_files_from_dir(os.path.join(self.data_path, 'anomaly'), label=1)
            
            all_files = normal_files + anomaly_files
        else:
            if os.path.exists(os.path.join(self.data_path, 'data.json')):
                with open(os.path.join(self.data_path, 'data.json'), 'r') as f:
                    data = json.load(f)
                    all_files = [(item['logs'], item['label']) for item in data]
            else:
                all_files = self._generate_synthetic_data()
        
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * self.train_ratio)
        if self.mode == 'train':
            selected_files = all_files[:split_idx]
        else:
            selected_files = all_files[split_idx:]
        
        for logs, label in selected_files:
            self.samples.append(logs)
            self.labels.append(label)
    
    def _load_files_from_dir(self, dir_path: str, label: int) -> List[Tuple[List[str], int]]:
        files = []
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.log') or filename.endswith('.txt'):
                    file_path = os.path.join(dir_path, filename)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        logs = f.readlines()
                    files.append((logs, label))
        return files
    
    def _generate_synthetic_data(self) -> List[Tuple[List[str], int]]:
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
            if i < 70:
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
            else:
                logs = []
                for j in range(50):
                    if random.random() < 0.3:
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