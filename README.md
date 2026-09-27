# DentalProject

口腔医学三维图像智能分割与临床影像分析系统

## 项目结构

```
project/
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 预处理后数据
├── experiments/
│   ├── cbct_seg/         # CBCT 三维分割
│   │   └── dataset.py    # CBCT 数据加载器
│   ├── pointcloud_seg/    # 口扫点云分割
│   │   ├── dataset.py    # 点云数据加载器
│   │   ├── model.py      # PointNet 模型
│   │   ├── preprocess.py  # 点云预处理
│   │   └── train.py       # 训练脚本
│   └── registration/      # 配准融合
├── agent/                # 临床影像分析 Agent
│   ├── app.py            # Gradio 界面
│   └── llm_agent.py      # LLM 核心
├── checkpoints/          # 模型权重
├── results/              # 实验结果
└── deployment/           # C++ 部署
```

## 快速开始

```bash
# 激活环境
source /root/miniconda3/bin/activate

# 运行点云训练
cd experiments/pointcloud_seg
python train.py

# 启动 Agent demo
cd agent
python app.py
```

## 技术栈

- PyTorch 2.8 + CUDA 12.8
- MONAI, Open3D, nibabel, SimpleITK
- PointNet / Point Transformer
- Gradio (Agent interface)

