import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Dict, Tuple
import random
import numpy as np
import re


class Single2SingleDataset(Dataset):
    def __init__(self, data_path: str, mode: str = 'train', train_ratio: float = 0.8, sample_ratio: float = 1.0):
        self.data_path = data_path
        self.mode = mode
        self.train_ratio = train_ratio
        self.sample_ratio = sample_ratio  # 数据采样比例
        
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
        """加载数据，支持多种数据格式，在文件读取阶段进行采样以节省内存"""
        single2single_path = os.path.join(self.data_path, 'Single2Single')
        
        # 优先从Single2Single目录加载实际数据集
        if os.path.exists(single2single_path):
            all_files = self._load_single2single_data_sampled(single2single_path)
        # 备选方案：从normal/anomaly目录加载
        elif os.path.exists(os.path.join(self.data_path, 'normal')):
            normal_files = self._load_files_from_dir_sampled(os.path.join(self.data_path, 'normal'), label=0)
            anomaly_files = self._load_files_from_dir_sampled(os.path.join(self.data_path, 'anomaly'), label=1)
            all_files = normal_files + anomaly_files
        # 从JSON文件加载
        elif os.path.exists(os.path.join(self.data_path, 'data.json')):
            with open(os.path.join(self.data_path, 'data.json'), 'r') as f:
                data = json.load(f)
                # 在JSON加载时进行采样
                if self.sample_ratio < 1.0:
                    sample_size = int(len(data) * self.sample_ratio)
                    data = random.sample(data, sample_size)
                    print(f"从JSON采样 {self.sample_ratio*100:.1f}% 的数据，共 {len(data)} 个样本")
                all_files = [(item['logs'], item['label']) for item in data]
        # 生成合成数据
        else:
            print("未找到实际数据集，生成合成数据用于演示...")
            all_files = self._generate_synthetic_data()
        
        if not all_files:
            print("警告：未找到任何数据文件，生成合成数据...")
            all_files = self._generate_synthetic_data()
        
        random.shuffle(all_files)
        
        # 确保有足够的样本进行训练/验证分割
        min_samples_needed = 2  # 至少需要2个样本才能分割
        if len(all_files) < min_samples_needed:
            print(f"警告：样本数量过少（{len(all_files)}个），生成额外合成数据以确保训练...")
            additional_synthetic = self._generate_synthetic_data()
            all_files.extend(additional_synthetic[:min_samples_needed - len(all_files) + 2])
            random.shuffle(all_files)
        
        # 分割训练和验证数据，确保两个集合都至少有1个样本
        total_samples = len(all_files)
        if total_samples == 1:
            # 如果只有1个样本，复制一份确保训练和验证都有数据
            if self.mode == 'train':
                selected_files = all_files
            else:
                selected_files = all_files  # 验证集也使用同一个样本
        else:
            # 确保训练集至少有1个样本，验证集也至少有1个样本
            min_train_samples = max(1, int(total_samples * self.train_ratio))
            min_val_samples = max(1, total_samples - min_train_samples)
            
            # 如果按比例分割会导致某个集合为空，则调整分割点
            if min_train_samples >= total_samples:
                min_train_samples = total_samples - 1
                min_val_samples = 1
            
            if self.mode == 'train':
                selected_files = all_files[:min_train_samples]
            else:
                selected_files = all_files[min_train_samples:]
        
        for logs, label in selected_files:
            self.samples.append(logs)
            self.labels.append(label)
        
        print(f"加载{self.mode}数据: {len(self.samples)}个样本")
        
        # 如果仍然没有样本，说明有严重问题
        if len(self.samples) == 0:
            raise ValueError(f"无法加载任何{self.mode}数据！请检查数据路径和采样设置。")
    
    def _load_single2single_data_sampled(self, single2single_path: str) -> List[Tuple[List[str], int]]:
        """加载Single2Single数据集，按异常类型分层采样"""
        all_files = []
        
        # 按异常类型分组收集文件
        files_by_type = {}
        
        # 遍历所有异常类型目录
        for item_name in os.listdir(single2single_path):
            item_path = os.path.join(single2single_path, item_name)
            
            # 跳过非目录文件
            if not os.path.isdir(item_path):
                continue
            
            # 确定标签
            if item_name in self.anomaly_types:
                label = 1  # 异常
                anomaly_type = item_name
            else:
                label = 0  # 正常
                anomaly_type = 'normal'
            
            # 收集该类型的所有文件
            type_files = []
            for filename in os.listdir(item_path):
                file_path = os.path.join(item_path, filename)
                if os.path.isdir(file_path):
                    continue
                if filename.endswith(('.log', '.txt')) or not '.' in filename:
                    type_files.append((file_path, label, anomaly_type))
            
            if type_files:
                files_by_type[anomaly_type] = type_files
        
        # 检查单个文件
        single_files_by_type = {}
        for item_name in os.listdir(single2single_path):
            item_path = os.path.join(single2single_path, item_name)
            
            if os.path.isfile(item_path) and not item_name.endswith('.tar.gz'):
                # 根据文件名确定异常类型
                anomaly_type = None
                for atype in self.anomaly_types.keys():
                    if atype in item_name:
                        anomaly_type = atype
                        label = 1
                        break
                
                if anomaly_type is None:
                    anomaly_type = 'normal'
                    label = 0
                
                if anomaly_type not in single_files_by_type:
                    single_files_by_type[anomaly_type] = []
                single_files_by_type[anomaly_type].append((item_path, label, anomaly_type))
        
        # 合并目录文件和单个文件
        for anomaly_type, type_files in single_files_by_type.items():
            if anomaly_type in files_by_type:
                files_by_type[anomaly_type].extend(type_files)
            else:
                files_by_type[anomaly_type] = type_files
        
        print(f"发现 {len(files_by_type)} 种异常类型:")
        for anomaly_type, type_files in files_by_type.items():
            print(f"  {anomaly_type}: {len(type_files)} 个文件")
        
        # 对每种异常类型进行采样
        selected_files = []
        total_original_files = 0
        total_sampled_files = 0
        
        for anomaly_type, type_files in files_by_type.items():
            total_original_files += len(type_files)
            
            if self.sample_ratio < 1.0:
                # 每种类型至少选择1个文件，确保所有类型都有代表
                sample_size = max(1, int(len(type_files) * self.sample_ratio))
                if sample_size > len(type_files):
                    sample_size = len(type_files)
                sampled_files = random.sample(type_files, sample_size)
                print(f"  {anomaly_type}: 从 {len(type_files)} 个文件中采样 {len(sampled_files)} 个 ({sample_size/len(type_files)*100:.1f}%)")
            else:
                sampled_files = type_files
                print(f"  {anomaly_type}: 使用全部 {len(type_files)} 个文件")
            
            selected_files.extend(sampled_files)
            total_sampled_files += len(sampled_files)
        
        print(f"总计：从 {total_original_files} 个文件中采样了 {total_sampled_files} 个文件 ({total_sampled_files/total_original_files*100:.1f}%)")
        
        # 只读取选中的文件
        for file_path, label, anomaly_type in selected_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    logs = f.readlines()
                if logs:  # 确保文件不为空
                    all_files.append((logs, label))
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
        
        print(f"成功加载 {len(all_files)} 个文件的数据")
        
        return all_files
    
    def _load_files_from_dir_sampled(self, dir_path: str, label: int) -> List[Tuple[List[str], int]]:
        """从目录加载日志文件，在文件读取阶段进行采样"""
        files = []
        if not os.path.exists(dir_path):
            return files
        
        # 首先收集所有可用文件路径
        available_files = []
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            # 跳过子目录
            if os.path.isdir(file_path):
                continue
            
            # 只处理日志文件
            if filename.endswith(('.log', '.txt')) or not '.' in filename:
                available_files.append(file_path)
        
        # 根据采样比例选择文件，但至少选择1个文件
        if self.sample_ratio < 1.0:
            sample_size = max(1, int(len(available_files) * self.sample_ratio))
            if sample_size > len(available_files):
                sample_size = len(available_files)
            selected_files = random.sample(available_files, sample_size)
            print(f"从目录 {dir_path} 的 {len(available_files)} 个文件中采样 {len(selected_files)} 个")
        else:
            selected_files = available_files
        
        # 只读取选中的文件
        for file_path in selected_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    logs = f.readlines()
                if logs:  # 确保文件不为空
                    files.append((logs, label))
            except Exception as e:
                print(f"读取文件 {os.path.basename(file_path)} 时出错: {e}")
        
        return files
    
    def _generate_synthetic_data(self) -> List[Tuple[List[str], int]]:
        """生成合成数据用于演示"""
        synthetic_data = []
        
        # 根据采样比例调整生成的数据量
        base_samples = 100
        if self.sample_ratio < 1.0:
            total_samples = max(10, int(base_samples * self.sample_ratio))
            print(f"生成 {total_samples} 个合成样本 (采样比例: {self.sample_ratio*100:.1f}%)")
        else:
            total_samples = base_samples
        
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
        
        for i in range(total_samples):
            if i < int(total_samples * 0.7):  # 70%正常数据
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
    def __init__(self, data_path: str, num_nodes: int = 1, mode: str = 'train', train_ratio: float = 0.8, sample_ratio: float = 1.0):
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.mode = mode
        self.train_ratio = train_ratio
        self.sample_ratio = sample_ratio
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """为多节点配置加载数据"""
        base_dataset = Single2SingleDataset(self.data_path, self.mode, self.train_ratio, self.sample_ratio)
        
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
                     train_ratio: float = 0.8,
                     sample_ratio: float = 1.0) -> DataLoader:
    """创建数据加载器"""
    
    if num_nodes == 1:
        dataset = Single2SingleDataset(data_path, mode, train_ratio, sample_ratio)
    else:
        dataset = MultiNodeDataset(data_path, num_nodes, mode, train_ratio, sample_ratio)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=0,
        collate_fn=lambda x: x
    )