# privacy_gnn_rl_project

privacy_gnn_rl_project/
├── configs/                     # YAML configs for training setups
│   └── baseline.yaml           # Baseline config (GCN + fixed ε or RL)
│
├── data/                        # Data loading & preprocessing
│   └── flickr_loader.py        # Loads SNAP Flickr dataset as PyG format
│
├── models/                      # Core model components
│   ├── gnn.py                  # GCN / GAT encoder
│   ├── dp_engine.py            # DP mechanism (Laplace)
│   ├── policy_mlp.py           # RL controller (MLP version)
│   ├── policy_transformer.py   # (Optional) RL controller (Transformer)
│   └── text_encoder.py         # (Future) Placeholder for multimodal input
│
├── core/                        # Training pipelines
│   ├── trainer_baseline.py     # Standard GCN training
│   ├── trainer_dp_static.py    # Fixed ε DP training
│   ├── trainer_dp_rl.py        # RL-based ε allocation training
│   └── trainer_dp_rule.py      # (Optional) Rule-based policy
│
├── experiments/                 # Experiment scripts for entry point
│   ├── run_baseline.py         # GCN without DP
│   ├── run_dp_static.py        # GCN with fixed DP
│   ├── run_dp_rl.py            # GCN with RL ε control
│   └── run_transformer_policy.py # RL ε control with transformer
│
├── utils/                       # Utilities
│   ├── visualizer.py           # ε heatmap / network visualization
│   ├── evaluator.py            # Accuracy, F1, privacy stats
│   └── attack_mia.py           # Membership Inference Attack (optional)
│
├── logs/                        # Log & output files (tensorboard, wandb)
├── README.md                    # Project overview & run instructions
├── requirements.txt             # Python dependencies
└── run.sh                       # One-click run shell script





    +-----------------------------+
    |        Flickr Dataset       |
    |  (Node features + edges)   |
    +-----------------------------+
                     ↓
         data/flickr_loader.py
                     ↓
        +---------------------+
        |    GNN Encoder      |
        |  (e.g., GCN, GAT)   |
        +---------------------+
                     ↓
        +-----------------------------+
        |  Privacy Controller Policy   | ← MLP / Transformer (selects ε)
        +-----------------------------+
                     ↓
         +------------------------+
         |   DP Engine (Laplace)  |
         +------------------------+
                     ↓
         +------------------------+
         |   Noised GNN Output    |
         +------------------------+
                     ↓
         +------------------------+
         |     Task Prediction    |
         |  (e.g., Node Labeling) |
         +------------------------+
                     ↓
         +------------------------+
         |    Evaluation & MIA    |
         +------------------------+