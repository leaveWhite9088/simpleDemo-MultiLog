"""
日志解析与分组模块
实现论文《Multivariate Log-based Anomaly Detection for Distributed Database》中的日志预处理

该模块实现论文Section 3.1.1中描述的两个关键步骤：
1. 日志解析 - 使用Drain3算法将非结构化日志解析为结构化事件
2. 日志分组 - 采用固定时间窗口方法将事件序列切分为日志组

对应论文Section 3.1.1 - Log Parsing and Grouping
"""

import re
from typing import List, Dict, Tuple
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
from datetime import datetime, timedelta


class LogParser:
    """
    日志解析器
    
    论文要求：使用Drain3算法将非结构化的原始日志文本解析为结构化的日志事件
    
    功能：
    1. 解析原始日志行，提取时间戳和内容
    2. 使用Drain3算法识别日志模板
    3. 为每个日志事件分配唯一的事件ID
    
    对应论文Section 3.1.1 - Log Parsing
    输出：结构化日志事件列表
    """
    def __init__(self, config_file: str = None):
        # 初始化Drain3解析器
        # 论文要求：使用Drain3算法进行日志解析
        if config_file:
            config = TemplateMinerConfig()
            config.load(config_file)
            self.drain_parser = TemplateMiner(config=config)
        else:
            self.drain_parser = TemplateMiner()
        
        # 事件ID映射表：template_id -> event_id
        self.event_id_map = {}
        self.event_templates = {}
        self.event_counter = 0
    
    def parse_log_line(self, log_line: str) -> Tuple[str, str, datetime]:
        """
        解析单行日志
        
        处理流程：
        1. 提取时间戳信息
        2. 使用Drain3解析日志模板
        3. 分配事件ID
        
        输入：log_line - 原始日志行
        输出：(event_id, content, timestamp)
        
        论文描述：将非结构化日志解析为结构化事件
        """
        # 步骤1：时间戳提取
        # 支持标准时间格式：YYYY-MM-DD HH:MM:SS
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        match = re.search(timestamp_pattern, log_line)
        
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            content = log_line[match.end():].strip()
        else:
            # 如果无法提取时间戳，使用当前时间
            timestamp = datetime.now()
            content = log_line.strip()
        
        # 步骤2：Drain3日志解析
        # 论文要求：使用Drain3算法识别日志模板
        result = self.drain_parser.add_log_message(content)
        template_id = result.cluster_id
        
        # 步骤3：事件ID分配
        # 为每个唯一的日志模板分配连续的事件ID
        if template_id not in self.event_id_map:
            self.event_id_map[template_id] = self.event_counter
            self.event_templates[self.event_counter] = result.get_template()
            self.event_counter += 1
        
        event_id = self.event_id_map[template_id]
        
        return str(event_id), content, timestamp
    
    def parse_logs(self, log_lines: List[str]) -> List[Dict]:
        """
        批量日志解析
        
        输入：log_lines - 原始日志文本列表
        输出：解析后的结构化事件列表
        
        每个事件包含：
        - event_id: 事件标识符
        - content: 日志内容
        - timestamp: 时间戳
        
        对应论文中的日志解析阶段
        """
        parsed_logs = []
        for line in log_lines:
            if line.strip():  # 跳过空行
                event_id, content, timestamp = self.parse_log_line(line)
                parsed_logs.append({
                    'event_id': event_id,
                    'content': content,
                    'timestamp': timestamp
                })
        return parsed_logs


class LogGrouper:
    """
    日志分组器
    
    论文要求：采用固定时间窗口方法将解析后的日志事件序列切分为多个固定长度的日志组
    
    分组策略：
    - 时间窗口大小：论文建议5秒
    - 分组方法：滑动窗口或固定窗口
    
    对应论文Section 3.1.1 - Log Grouping
    输出：日志组列表 [S_1, S_2, ..., S_k]
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size  # 时间窗口大小（秒）
    
    def group_logs_by_window(self, parsed_logs: List[Dict]) -> List[List[Dict]]:
        """
        固定时间窗口分组
        
        论文描述：将解析后的日志事件序列切分为多个固定长度的日志组S_j
        
        算法：
        1. 按时间戳排序所有日志事件
        2. 使用固定时间窗口（5秒）进行分组
        3. 每个时间窗口内的事件构成一个日志组
        
        输入：parsed_logs - 解析后的日志事件列表
        输出：日志组列表，每个组包含一个时间窗口内的所有事件
        
        对应论文中的"fixed time window"分组策略
        """
        if not parsed_logs:
            return []
        
        # 步骤1：按时间戳排序
        parsed_logs.sort(key=lambda x: x['timestamp'])
        
        # 步骤2：固定时间窗口分组
        groups = []
        start_time = parsed_logs[0]['timestamp']
        current_group = []
        
        for log in parsed_logs:
            # 检查是否在当前时间窗口内
            if log['timestamp'] < start_time + timedelta(seconds=self.window_size):
                current_group.append(log)
            else:
                # 当前窗口结束，开始新窗口
                if current_group:
                    groups.append(current_group)
                    
                # 开始新的时间窗口
                start_time = log['timestamp']
                current_group = [log]
        
        # 添加最后一个组
        if current_group:
            groups.append(current_group)
        
        return groups