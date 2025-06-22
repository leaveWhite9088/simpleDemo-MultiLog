import torch
from data.dataset import Single2SingleDataset, create_dataloader
from MultiLog.log_parser import LogParser, LogGrouper
import os

def test_log_parsing():
    """测试日志解析功能"""
    print("=== 测试日志解析 ===")
    
    # 测试不同格式的日志
    test_logs = [
        "- 2023-11-18 08:46:14,674 [main] INFO  o.a.i.c.c.ConfigNodeDescriptor:105 - Start to read config file",
        "1700297194.920928",
        "Msg: The statement is executed successfully.",
        "437223316f62   cluster-iotdb:1.2.2   \"/bin/sh -c '/iotdb/…\"   37 seconds ago   Up 36 seconds",
        "start inject compaction",
        "end inject compaction"
    ]
    
    parser = LogParser()
    for log in test_logs:
        try:
            event_id, content, timestamp = parser.parse_log_line(log)
            print(f"Log: {log[:50]}...")
            print(f"  Event ID: {event_id}, Timestamp: {timestamp}")
            print(f"  Content: {content[:50]}...")
        except Exception as e:
            print(f"Error parsing log: {e}")
        print()

def test_dataset_loading():
    """测试数据集加载"""
    print("\n=== 测试数据集加载 ===")
    
    data_path = './data'
    
    # 测试单节点数据集
    try:
        dataset = Single2SingleDataset(data_path, mode='train', sample_ratio=0.01)
        print(f"训练集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 获取第一个样本
            logs, label = dataset[0]
            print(f"第一个样本:")
            print(f"  日志条数: {len(logs)}")
            print(f"  标签: {label}")
            print(f"  前3条日志:")
            for i, log in enumerate(logs[:3]):
                print(f"    {i+1}: {log.strip()[:80]}...")
    except Exception as e:
        print(f"加载数据集时出错: {e}")

def test_dataloader():
    """测试DataLoader"""
    print("\n=== 测试DataLoader ===")
    
    data_path = './data'
    
    try:
        # 创建训练数据加载器
        train_loader = create_dataloader(
            data_path,
            batch_size=2,
            num_nodes=1,
            mode='train',
            sample_ratio=0.01
        )
        
        print(f"DataLoader创建成功")
        
        # 测试获取一个批次
        for batch in train_loader:
            print(f"批次大小: {len(batch)}")
            for i, (logs, label) in enumerate(batch):
                print(f"  样本{i+1}: 日志数={len(logs)}, 标签={label}")
            break  # 只测试第一个批次
            
    except Exception as e:
        print(f"创建DataLoader时出错: {e}")

def test_log_grouping():
    """测试日志分组"""
    print("\n=== 测试日志分组 ===")
    
    # 读取一个实际的日志文件
    log_file = './data/Single2Single/compaction_continue/label1.log'
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = f.readlines()[:50]  # 只读取前50行
            
            parser = LogParser()
            grouper = LogGrouper(window_size=5)
            
            # 解析日志
            parsed_logs = parser.parse_logs(log_lines)
            print(f"解析了 {len(parsed_logs)} 条日志")
            
            # 分组
            log_groups = grouper.group_logs_by_window(parsed_logs)
            print(f"分成了 {len(log_groups)} 个日志组")
            
            for i, group in enumerate(log_groups[:3]):
                print(f"  组{i+1}: 包含 {len(group)} 条日志")
                
        except Exception as e:
            print(f"处理日志文件时出错: {e}")
    else:
        print(f"日志文件不存在: {log_file}")

if __name__ == "__main__":
    test_log_parsing()
    test_dataset_loading()
    test_dataloader()
    test_log_grouping()