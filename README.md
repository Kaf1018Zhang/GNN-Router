# GNN-Router
Personal Project

## 📁 Project Structure

```
DP-GNN/
│
├── datasets/
│   ├── xxx_loader.py                # Loader of datasets like PROTEINS, ENZYMES, etc.
│   └── loader_factory.py            # Load all avaliable datasets like PROTEINS, ENZYMES, etc.
│
├── modules/
│   ├── controller.py                # RL controller logic
│   ├── composed_model.py            # Compose full GNN model
│   └── composable_blocks.py         # Encoder, pooling, readout, augmentation blocks
│
├── utils/
│   ├── evaluator.py                 # LightningModule evaluator
│   └── visualize.py                 # Visualization for episodes and overall
│
├── experiments/
│   ├── train_random_controller.py   # Random strategy sampling
│   ├── train_rl_controller.py       # RL-based architecture search
│   └── deploy_controller.py         # Evaluate saved best strategy
│
├── vis/
│   ├── episode/                     # Per-episode loss/accuracy plots
│   └── overall/                     # Val acc/loss across episodes
│
├── best_strategy.json               # Output from RL training
└── README.md                        # Main documentation
```
