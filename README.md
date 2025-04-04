# GNN-Router
Personal Project

📁 DP-GNN
├── datasets/
│   └── loader_factory.py             # 数据加载器：支持 PROTEINS, ENZYMES 等多图分类数据集
│
├── modules/
│   ├── controller.py                 # 强化学习控制器，用于策略生成
│   ├── composed_model.py             # 模块化 GNN 组合模型入口
│   ├── composable_blocks.py          # 可组合的编码器/池化/读出/增强模块
│
├── utils/
│   ├── evaluator.py                  # PyTorch Lightning 模型封装 & 训练/验证逻辑
│   ├── visualize.py                  # 可视化模块，支持 episode 和 overall 分支
│
├── experiments/
│   ├── train_controller.py           # 固定策略训练 baseline（controller 被动测试）
│   ├── train_random_controller.py    # 随机策略 baseline，用于对比强化学习效果
│   ├── train_rl_controller.py        # 强化学习策略搜索主程序
│   ├── deploy_controller.py          # 使用已选最优策略部署并测试性能
│
├── vis/
│   ├── episode/                      # 每轮 RL 的训练过程图（train/val loss）
│   └── overall/                      # 所有 episode 过程中的 val acc 与损失可视化
│
├── best_strategy.json                # RL 训练后保存的最优策略组合
└── README.md                         # 项目说明文档（建议将本结构粘贴进此处）
