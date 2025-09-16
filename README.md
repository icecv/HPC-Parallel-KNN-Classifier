# KNN算法K-Fold交叉验证并行化实现

这是一个基于OpenMP和MPI的K近邻(KNN)算法并行化实现项目，支持K-Fold交叉验证，适用于高性能计算环境。

## 项目概述

本项目实现了KNN算法的两种并行化方案：
- **OpenMP并行化**：针对算法内部计算的线程级并行
- **MPI并行化**：针对K-Fold交叉验证的进程级并行
- **混合并行化**：结合OpenMP和MPI的两级并行

## 文件结构

```
├── k-folds.c           # 串行K-Fold交叉验证实现
├── mpi-k-folds.c       # MPI分布式K-Fold交叉验证实现
├── knnomp.c            # OpenMP并行化KNN算法核心
├── file-reader.c       # CSV文件读写工具
├── Makefile           # 编译配置文件
├── runmpi.sh          # SLURM集群作业提交脚本
└── README.md          # 项目说明文档
```

## 功能特性

### KNN算法实现
- 支持多维特征空间的欧几里得距离计算
- 实现K近邻搜索和分类预测
- 包含平局处理机制（选择最近邻的类别）
- OpenMP并行化距离计算过程

### K-Fold交叉验证
- 自动数据分割和训练/测试集生成
- 支持不等长fold处理
- 计算每个fold的分类准确率
- 输出平均准确率统计

### 并行化策略
- **OpenMP**：并行化KNN算法中的距离计算循环
- **MPI**：将不同fold分配给不同进程并行处理
- **内存优化**：使用连续内存和高效的memcpy操作

## 编译方式

本项目支持GNU和Intel编译器：

```bash
# 编译所有版本
make all

# 单独编译
make gccnearly      # GCC串行版本
make iccnearly      # Intel ICC串行版本
make gcccomplete    # GCC+MPI分布式版本
make icccomplete    # Intel ICC+MPI分布式版本

# 清理编译文件
make clean
```

## 运行方式

### 串行版本
```bash
./k-folds-gcc <数据文件.csv> <输出文件.csv> <k值> <fold数量>
```

例如：
```bash
./k-folds-gcc asteroids.csv output.csv 3 10
```

### 分布式版本
```bash
mpirun -np <进程数> ./k-folds-complete-gcc <数据文件.csv> <输出文件.csv> <k值> <fold数量>
```

例如：
```bash
mpirun -np 4 ./k-folds-complete-gcc asteroids.csv output.csv 3 10
```

### 集群环境运行
```bash
# 提交到SLURM调度系统
sbatch runmpi.sh
```

## 数据格式要求

输入的CSV文件需要满足以下格式：
- 数值型特征数据（浮点数）
- 最后一列为整数类标签（从0开始）
- 无表头行
- 逗号分隔

示例数据格式：
```csv
0.5,5.3,12.3,0
0.75,2.0,8.7,1
0.6,4.9,12.1,0
0.9,1.8,8.1,1
```

## 性能测试

项目包含自动化性能测试脚本，测试内容包括：

### OpenMP性能测试
- 线程数：1, 2, 4, 8
- 测量运行时间和加速比
- 计算并行效率

### MPI性能测试
- 进程数：1, 2, 4, 8, 16, 32
- 强扩展性分析
- 多节点性能评估

### 测试参数
- 数据集：your_data.csv
- K值：3
- Fold数：10

## 算法原理

### K近邻算法
1. 计算测试点与所有训练点的欧几里得距离
2. 选择距离最小的K个近邻点
3. 统计K个近邻点的类别分布
4. 选择出现频率最高的类别作为预测结果
5. 平局情况下选择距离最近点的类别

### K-Fold交叉验证
1. 将数据集划分为K个大小相等的子集
2. 依次选择一个子集作为测试集，其余作为训练集
3. 在每个训练集上训练模型，在对应测试集上评估
4. 计算K次评估的平均准确率

## 并行化设计

### OpenMP并行化
```c
#pragma omp parallel for
for (int i = 0; i < testpointnum; i++) {
    // 并行计算每个测试点的KNN预测
}
```

### MPI并行化
- 主进程读取数据并广播给所有进程
- 每个进程处理特定的fold子集
- 使用MPI_Reduce聚合各进程的准确率结果

## 环境要求

### 编译环境
- GCC 4.9+ 或 Intel ICC
- OpenMP支持
- MPI库（OpenMPI或Intel MPI）

### 运行环境
- Linux系统
- SLURM作业调度系统（可选）
- 多核CPU和/或多节点集群

## 输出结果

程序输出包含：
- 每个fold的分类准确率
- 平均准确率
- 运行时间统计（通过脚本测试）

输出文件格式：
```
fold0_accuracy,fold1_accuracy,...,average_accuracy
```

