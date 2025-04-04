# GNN-Router
Personal Project

📁 DP-GNN
├── datasets/
│   └── loader_factory.py              # Dataset loader for PROTEINS, ENZYMES, etc.
│
├── modules/
│   ├── controller.py                  # RL-based controller for sampling strategies
│   ├── composed_model.py              # Entry point for composed GNN model
│   ├── composable_blocks.py           # Encoder, pooling, readout, and augmentation blocks
│
├── utils/
│   ├── evaluator.py                   # Lightning evaluator with training/validation logic
│   ├── visualize.py                   # Visualizer for per-episode and overall plots
│
├── experiments/
│   ├── train_controller.py            # Train with fixed strategies (non-RL baseline)
│   ├── train_random_controller.py     # Train with randomly sampled strategies
│   ├── train_rl_controller.py         # Train controller with RL strategy search
│   ├── deploy_controller.py           # Load best strategy and run final evaluation
│
├── vis/
│   ├── episode/                       # Per-episode visualizations (train/val loss)
│   └── overall/                       # Val accuracy & reward curves across episodes
│
├── best_strategy.json                 # Saved best strategy from RL training
└── README.md                          # Documentation (insert this structure section here)
