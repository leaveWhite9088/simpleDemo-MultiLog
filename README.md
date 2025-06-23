# MultiLog: 分布式数据库多元日志异常检测

基于KDD '24论文《Multivariate Log-based Anomaly Detection for Distributed Database》的Demo实现。

❗特别注意：本demo项目仅用于北京大学软件与微电子学院夏令营考核，如有生产或研究需要，请考虑参考原论文

## 📋 项目简介

MultiLog 是一个用于分布式数据库日志异常检测的多变量深度学习模型。该模型采用一个两阶段的监督学习框架，旨在准确识别分布式系统中的各类异常。

- **第一阶段 (独立评估)**：为每个数据库节点独立分析日志，提取序列、量化、语义等多维特征，并生成初步的异常概率。
- **第二阶段 (集群分类)**：智能地聚合所有节点的概率信息，通过一个元分类器对整个集群的状态做出最终的综合判定。

## ✨ 主要特性

- **两阶段检测架构**：将节点独立分析与集群全局决策解耦，精准度高。
- **多维特征嵌入**：融合日志的序列、量化及语义信息，全面捕捉异常特征。
- **多场景支持**：支持单节点、多节点、单异常、多异常等多种复杂的实验场景。
- **Demo可视化**：内置训练历史、混淆矩阵、ROC/PR曲线等多种可视化工具。
- **GPU加速**：支持CUDA加速，并能自动检测可用设备，大幅提升训练效率。

## 🏗️ 模型架构

### 第一阶段：独立评估（Standalone Estimation）

1. 日志解析 (LogParser)
   - 使用 **Drain3** 算法解析非结构化日志，提取日志模板和参数。
2. 三重特征嵌入 (Embeddings)
   - **序列嵌入**：捕获日志事件的序列模式。
   - **量化嵌入**：统计时间窗口内各事件的频率。
   - **语义嵌入**：使用预训练的 **FastText** 模型提取日志模板的深层语义信息。
3. 信息增强 (Enhancement)
   - 采用 **LSTM + Self-Attention** 机制，对三种嵌入特征进行加权和增强，捕捉长距离依赖关系。
4. 异常概率预测
   - 输出每个日志时间窗口的异常概率。

### 第二阶段：集群分类（Cluster Classifier）

1. 概率标准化 (AutoEncoder)
   - 利用自编码器处理来自不同节点、长度不一的概率序列，并将其映射到固定维度的隐向量。
2. 元分类器 (MetaClassifier)
   - 将所有节点的标准化隐向量进行拼接，送入一个最终分类器，输出整个集群的 Normal/Anomaly 判定。

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少 8GB RAM

### 2. 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd simpleDemo-MultiLog

# 2. 创建并激活虚拟环境（强烈推荐）
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 3. 安装依赖
# 推荐使用清华镜像源加速下载
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 数据集准备

#### 数据集下载

本项目仅支持论文中提到的Single2Single数据集配置（压缩包5G，解压后120G左右）：

- **Single2Single**: [下载链接](https://zenodo.org/records/11496301/files/Single2Single.tar.gz)

#### 目录结构

将下载的数据集解压到 `./data/` 目录下，确保其结构如下：

```
data/
└── Single2Single/
    ├── cpu_continue/   # 异常类型目录
    │   └── *.log       # 日志文件
    ├── io_continue/
    │   └── *.log
    └── ... (其他异常类型目录)
```

#### 异常类型说明

数据集中共包含11种精心设计的异常类型（论文数据库仓库：[AIOps-LogDB/MultiLog-Dataset](https://github.com/AIOps-LogDB/MultiLog-Dataset?tab=readme-ov-file)）：

| 编号 | 异常类型                                     | 分类   | 描述                           |
| ---- | -------------------------------------------- | ------ | ------------------------------ |
| 1    | CPU饱和 (CPU Saturation)                     | 系统   | CPU计算资源耗尽                |
| 2    | IO饱和 (IO Saturation)                       | 系统   | I/O带宽严重占用                |
| 3    | 内存饱和 (Memory Saturation)                 | 系统   | 内存资源不足                   |
| 4    | 网络带宽限制 (Network Bandwidth Limited)     | 系统   | 节点间网络带宽受限             |
| 5    | 网络分区 (Network Partition)                 | 系统   | 节点间出现网络分区             |
| 6    | 机器宕机 (Machine Down)                      | 系统   | 运行时服务器宕机               |
| 7    | 慢查询伴随 (Accompanying Slow Query)         | 数据库 | 查询负载过重                   |
| 8    | 导出操作 (Export Operations)                 | 数据库 | 数据备份到外部                 |
| 9    | 导入操作 (Import Operations)                 | 数据库 | 从外部导入数据                 |
| 10   | 资源密集压缩 (Resource-Intensive Compaction) | 数据库 | 压缩任务消耗大量系统资源       |
| 11   | 频繁磁盘刷新 (Overly Frequent Disk Flushes)  | 数据库 | 低间隔刷新操作导致频繁磁盘写入 |

## 🔧 使用方法

### 1. 模型训练

**基础训练 (使用1%数据进行快速验证):**

```bash
python main.py
```

**带实时可视化的训练:**

```bash
python plt/train_with_visualization.py --epochs 10 --sample_ratio 0.01
```

**自定义训练参数:**

```bash
python plt/train_with_visualization.py \
    --epochs 20 \
    --sample_ratio 0.05 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --device cuda
```

### 2. 评估与可视化

#### 自动可视化
训练完成后，评估结果和可视化图表将自动生成并保存到 `plt/figures/` 目录。

#### 手动生成可视化

**使用真实训练结果:**

```bash
# 使用默认检查点路径
python plt/visualize_results.py

# 或指定检查点路径
python plt/visualize_results.py --checkpoint ./checkpoints/best_model.pth
```

**生成示例可视化图表:**
```bash
python plt/visualize_results.py --example
```

#### 可视化内容

- **查看位置**: `plt/figures/` 目录
- **包含内容**:
  - `training_history.png`: 训练/验证过程的损失和准确率曲线
  - `confusion_matrix.png`: 验证集上的混淆矩阵 (含准确率、精确率、召回率、F1分数)
  - `roc_curve.png`: ROC曲线与AUC值
  - `pr_curve.png`: Precision-Recall曲线与平均精度
  - `score_distribution.png`: 正常/异常样本的预测分数分布

### 3. 参数说明

| 参数              | 默认值          | 说明                                          |
| ----------------- | --------------- | --------------------------------------------- |
| `--data_path`     | `./data`        | 数据集根路径                                  |
| `--num_nodes`     | `1`             | 参与训练的节点数量                            |
| `--epochs`        | `20`            | 训练总轮数                                    |
| `--batch_size`    | `4`             | 每个批次的样本大小                            |
| `--learning_rate` | `0.001`         | 优化器的学习率                                |
| `--hidden_size`   | `128`           | LSTM隐藏层的大小                              |
| `--window_size`   | `5`             | 日志分组的时间窗口大小（单位：秒）            |
| `--sample_ratio`  | `0.05`          | 使用数据的比例，用于快速实验（1.0为全量数据） |
| `--device`        | `auto`          | 计算设备 (`auto`/`cuda`/`cpu`)                |
| `--save_path`     | `./checkpoints` | 训练模型的保存路径                            |

## 📖 论文参考

再次申明：本demo项目仅用于北京大学软件与微电子学院夏令营考核，如有生产或研究需要，请考虑参考原论文：

```
@inproceedings{multilog2024,
  title={Multivariate Log-based Anomaly Detection for Distributed Database},
  author={[Authors]},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```