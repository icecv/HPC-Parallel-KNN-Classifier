# KNN算法K-Fold交叉验证并行化实现

一个基于OpenMP和MPI的K近邻算法并行化实现，支持K-Fold交叉验证，适用于高性能计算环境。

## 项目概述

这是一个完整的KNN算法并行化项目，包含以下核心组件：
- **OpenMP并行化** - 算法内部的线程级并行优化
- **MPI分布式计算** - K-Fold交叉验证的进程级并行
- **混合并行模式** - 结合OpenMP和MPI的两级并行架构
- **高效内存管理** - 针对大数据集的内存优化策略

## 环境要求

- GCC 4.9+ 或 Intel ICC编译器
- OpenMP和MPI库支持
- Linux系统（推荐SLURM集群环境）

## 使用方式

**编译项目**
```bash
make all  # 编译所有版本
```

**运行程序**
```bash
# 串行版本
./k-folds-gcc 数据文件.csv 输出文件.csv k值 fold数量

# 分布式版本
mpirun -np 进程数 ./k-folds-complete-gcc 数据文件.csv 输出文件.csv k值 fold数量

# 集群提交
sbatch runmpi.sh
```

## 文件介绍

- **k-folds.c** - 串行K-Fold交叉验证主程序
- **mpi-k-folds.c** - MPI分布式版本主程序  
- **knnomp.c** - OpenMP并行化KNN算法核心
- **file-reader.c** - CSV文件读写工具库
- **Makefile** - 支持GCC和Intel编译器的编译配置
- **runmpi.sh** - SLURM集群自动化测试脚本

## 算法原理

**K近邻算法**
1. 计算测试点与所有训练点的欧几里得距离
2. 选择距离最小的K个近邻点
3. 统计K个近邻点的类别分布，选择出现频率最高的类别
4. 平局情况下选择距离最近点的类别

**K-Fold交叉验证**
1. 将数据集划分为K个大小相等的子集
2. 依次选择一个子集作为测试集，其余作为训练集
3. 在每个训练集上进行KNN预测，计算测试集准确率
4. 输出K次评估的平均准确率

## 并行化设计

**OpenMP并行化**
```c
#pragma omp parallel for
for (int i = 0; i < testpointnum; i++) {
    // 并行计算每个测试点的KNN预测
}
```

**MPI并行化**
- 主进程读取数据并广播给所有进程
- 每个进程处理不同的fold子集：`if (fold % size != rank) continue;`
- 使用MPI_Reduce聚合各进程的准确率结果

## 性能测试

项目包含自动化性能测试脚本`runmpi.sh`，测试内容包括：

**OpenMP性能测试**
- 线程数：1, 2, 4, 8
- 测量每种配置的运行时间
- 自动记录到`runtime_results_gcc.txt`和`runtime_results_icc.txt`

**MPI扩展性测试**
- 进程数：1, 2, 4, 8, 16, 32
- 使用最优OpenMP线程数（默认8线程）
- 支持多节点分布式测试

**测试数据集**
- 数据文件：asteroids.csv
- K值：3，Fold数：10
- 自动编译GCC和Intel ICC两个版本进行对比

## 数据格式

输入CSV文件要求：
- 数值型特征（前n-1列）
- 整数类标签（最后一列，从0开始）
- 无表头，逗号分隔

```csv
0.5,5.3,12.3,0
0.75,2.0,8.7,1
0.6,4.9,12.1,0
```
