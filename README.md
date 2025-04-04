# privacy_gnn_rl_project

Personal Project

composable-gnn-routing/
├── configs/                     # YAML configs for modular setups
│   └── strategy_gnn.yaml        # Defines GNN variants and RL options
│
├── datasets/                   # Dataset and loader
│   ├── proteins_loader.py       # PROTEINS dataset wrapper (PyG)
│   ├── enzymes_loader.py        # ENZYMES dataset wrapper (PyG)
│   └── loader_factory.py       # Interface to run all wappers
│
├── modules/                    # GNN building blocks
│   ├── gnn_base.py              # GCN, GAT, GIN modular blocks
│   ├── pooling.py               # mean, max, TopK pooling
│   └── routing_controller.py    # MLP / Transformer controller
│
├── core/                       # Training + RL
│   ├── trainer_strategy_rl.py   # Reinforcement learning training loop
│   ├── reward_utils.py          # reward design (acc - complexity)
│   └── module_executor.py       # Executes selected GNN module pipeline
│
├── experiments/               # Entry points
│   ├── deploy_controller
│   └── train_controller
│
├── utils/                      # Visualization & evaluation
│   ├── visualizer.py            # GNN path + accuracy heatmaps
│   └── evaluator.py             # Accuracy, loss tracking
│
├── logs/                       # TensorBoard, WandB, CSVs
├── README.md
├── requirements.txt
└── run.sh                      # One-click launcher