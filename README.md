# MultiLog: 分布式数据库多元日志异常检测

基于KDD '24论文《Multivariate Log-based Anomaly Detection for Distributed Database》的完整实现。

## 📋 项目简介

MultiLog是一个用于分布式数据库系统的多元日志异常检测模型，采用两阶段架构：
1. **独立评估阶段**：每个节点独立处理本地日志，输出异常概率
2. **集群分类阶段**：聚合所有节点的概率向量，进行全局异常检测

## 🚀 快速开始

### 环境要求
- Python 3.7+
- PyTorch 1.9.0+
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据集准备

本项目支持论文中提到的四种数据集配置：

#### 数据集下载
- **Single2Single**: [下载链接](https://zenodo.org/records/11496301/files/Single2Single.tar.gz)
- **Single2Multi**: [下载链接](https://zenodo.org/records/11496255/files/Single2Multi.tar.gz)  
- **Multi2Single**: [下载链接](https://zenodo.org/records/11483841/files/Multi2Single.tar.gz)
- **Multi2Multi**: [下载链接](https://zenodo.org/records/11468477/files/Multi2Multi.tar.gz)

#### 异常类型说明
数据集包含11种异常类型：

| 编号 | 异常类型 | 分类 | 描述 |
|------|----------|------|------|
| 1 | CPU饱和 | 系统 | CPU计算资源耗尽 |
| 2 | IO饱和 | 系统 | I/O带宽严重占用 |
| 3 | 内存饱和 | 系统 | 内存资源不足 |
| 4 | 网络带宽限制 | 系统 | 节点间网络带宽受限 |
| 5 | 网络分区 | 系统 | 节点间出现网络分区 |
| 6 | 机器宕机 | 系统 | 运行时服务器宕机 |
| 7 | 慢查询伴随 | 数据库 | 查询负载过重 |
| 8 | 导出操作 | 数据库 | 数据备份到外部 |
| 9 | 导入操作 | 数据库 | 从外部导入数据 |
| 10 | 资源密集压缩 | 数据库 | 压缩任务消耗大量系统资源 |
| 11 | 频繁磁盘刷新 | 数据库 | 低间隔刷新操作导致频繁磁盘写入 |

#### 数据目录结构
将下载的数据集解压到`./data/`目录下：

```
data/
├── Single2Single/
│   ├── [异常类型目录]/
│   └── Single2Single.tar.gz
├── dataset.py
└── __init__.py
```

## 🔧 使用方法

### 基本训练

**Single2Single配置（单节点）：**
```bash
python main.py --num_nodes 1 --epochs 20 --batch_size 4
```

**Multi2Multi配置（多节点）：**
```bash
python main.py --num_nodes 4 --epochs 50 --batch_size 8
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `./data` | 数据集路径 |
| `--num_nodes` | `1` | 节点数量 |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `4` | 批次大小 |
| `--learning_rate` | `0.001` | 学习率 |
| `--hidden_size` | `128` | LSTM隐藏层大小 |
| `--window_size` | `5` | 时间窗口大小（秒） |
| `--device` | `auto` | 计算设备（cuda/cpu） |
| `--save_path` | `./checkpoints` | 模型保存路径 |

## 🏗️ 项目结构

```
simpleDemo-MultiLog/
├── MultiLog/                    # 核心模型实现
│   ├── multilog.py             # 主模型
│   ├── standalone_estimation.py # 独立评估模块
│   ├── cluster_classifier.py   # 集群分类器
│   ├── embeddings.py          # 嵌入层实现
│   ├── enhancement.py         # 增强模块
│   └── log_parser.py          # 日志解析器
├── data/                       # 数据集和数据处理
│   ├── dataset.py             # 数据加载器
│   └── Single2Single/         # 数据集文件
├── main.py                    # 主程序入口
├── requirements.txt           # Python依赖
└── README.md                 # 项目说明
```

## 🧠 模型架构

### 第一阶段：独立评估（Standalone Estimation）
- **日志解析**：使用Drain3算法提取日志模板
- **三重嵌入**：
  - 序列嵌入（Sequential Embedding）
  - 量化嵌入（Quantitative Embedding）  
  - 语义嵌入（Semantic Embedding）
- **增强模块**：LSTM + Self-Attention机制

### 第二阶段：集群分类（Cluster Classifier）
- **概率标准化**：AutoEncoder统一不同长度的概率列表
- **元分类器**：基于拼接特征的全局分类器

## 📊 性能指标

模型在训练过程中会输出以下指标：
- **训练损失**（Train Loss）
- **训练准确率**（Train Accuracy）
- **验证损失**（Val Loss）
- **验证准确率**（Val Accuracy）

最佳模型会自动保存到`./checkpoints/best_model.pth`

## ⚠️ 注意事项

1. **FastText模型**：首次运行时需要下载预训练的FastText模型（cc.en.300.bin），如无法下载会使用随机嵌入
2. **内存使用**：多节点配置需要更多内存，建议根据机器配置调整`batch_size`
3. **GPU支持**：支持CUDA加速，会自动检测可用的GPU设备
4. **数据格式**：支持`.log`和`.txt`格式的日志文件

## 📖 论文引用

```bibtex
@inproceedings{multilog2024,
  title={Multivariate Log-based Anomaly Detection for Distributed Database},
  author={[Authors]},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目遵循MIT许可证，详见LICENSE文件。