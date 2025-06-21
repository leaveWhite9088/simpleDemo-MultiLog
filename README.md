# MultiLog Implementation

基于KDD '24论文实现的MultiLog模型，用于分布式数据库的多元日志异常检测。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将日志数据放在 `./data` 目录下：
- 正常日志：`./data/normal/` 
- 异常日志：`./data/anomaly/`

如果没有真实数据，模型会自动生成合成数据进行演示。

## 运行模型

Single2Single配置（单节点）：
```bash
python main.py --num_nodes 1 --epochs 20 --batch_size 4
```

主要参数：
- `--data_path`: 数据集路径（默认：./data）
- `--num_nodes`: 节点数量（Single2Single使用1）
- `--epochs`: 训练轮数（默认：20）
- `--batch_size`: 批次大小（默认：4）
- `--learning_rate`: 学习率（默认：0.001）
- `--device`: 设备（cuda/cpu）

## 模型结构

1. **Standalone Estimation**: 独立评估每个节点的日志
   - 日志解析（Drain3）
   - 三种嵌入：序列、量化、语义
   - LSTM + Self-Attention增强

2. **Cluster Classifier**: 集群级分类
   - AutoEncoder标准化概率列表
   - Meta-Classifier最终分类

## 注意事项

- FastText模型需要下载预训练模型（cc.en.300.bin），如果没有会使用随机嵌入
- 首次运行时会自动生成合成数据集用于演示