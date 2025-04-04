# experiments/train_rl_controller.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytorch_lightning as pl

from modules.rl_controller import MultiHeadRLController
from modules.composable_blocks import (
    GCNEncoder, GATEncoder, SAGEEncoder,
    pooling_mean, pooling_max, GlobalAttentionPooling,
    LinearReadout, MLPReadout, TransformerReadout,
    raw_features, spectral_features, virtual_node_features
)

from datasets.loader_factory import load_dataset
from utils.evaluator import Evaluator
from torch_geometric.loader import DataLoader
from core.reward_utils import compute_reward

import torch.nn as nn
import torch.nn.functional as F


# 1) 定义离散空间
encoder_opts = ["GCN", "GAT", "GraphSAGE"]
pooling_opts = ["mean", "max", "attention"]
readout_opts = ["linear", "mlp", "transformer"]
augment_opts = ["raw", "spectral", "virtual"]
hidden_dim_opts = [32, 64]
dropout_opts = [0.0, 0.5]
lr_opts = [1e-3, 1e-4]
bn_opts = [False, True]
temp_opts = [0.5, 1.0]


def build_model(action_dict, in_channels, out_channels):
    """
    根据actions选择具体Block并组合成可训练模型.
    """
    # Encoder
    if action_dict["encoder"] == "GCN":
        encoder = GCNEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])
    elif action_dict["encoder"] == "GAT":
        encoder = GATEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])
    else:
        encoder = SAGEEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])

    # Pooling
    if action_dict["pooling"] == "mean":
        pooling_fn = pooling_mean
    elif action_dict["pooling"] == "max":
        pooling_fn = pooling_max
    else:
        pooling_fn = GlobalAttentionPooling(action_dict["hidden_dim"])

    # Readout
    if action_dict["readout"] == "linear":
        readout = LinearReadout(action_dict["hidden_dim"], out_channels)
    elif action_dict["readout"] == "mlp":
        readout = MLPReadout(action_dict["hidden_dim"], out_channels, hidden_dim=action_dict["hidden_dim"])
    else:
        # transformer readout
        readout = TransformerReadout(d_model=action_dict["hidden_dim"], out_dim=out_channels)

    class ComposedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.pooling_fn = pooling_fn
            self.readout = readout
            self.dropout = action_dict["dropout"]

        def forward(self, x, edge_index, batch):
            # 1) 编码
            x = self.encoder(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 2) Pooling or pass to transformer
            #    - 如果 readout 是 transformer, 它需要 (x, batch) 形状
            #    - 如果 readout 是 linear/mlp, 它需要 pooled [batch_size, hidden_dim]
            if isinstance(self.readout, TransformerReadout):
                # 直接把 x, batch 交给 transformer, 它会在内部对每张图做CLS
                out = self.readout(x, batch)  # => [batch_size, out_dim]
            else:
                # 先全局pool => [batch_size, hidden_dim]
                if isinstance(self.pooling_fn, nn.Module):
                    # attention pooling
                    x_pooled = self.pooling_fn(x, batch)
                else:
                    x_pooled = self.pooling_fn(x, batch)
                out = self.readout(x_pooled)  # => [batch_size, out_dim]

            return out

    return ComposedModel()


def apply_feature_augment(data, augment_type):
    if augment_type == "raw":
        return raw_features(data)
    elif augment_type == "spectral":
        return spectral_features(data)
    else:
        return virtual_node_features(data)


def run_episode(dataset_name, controller, device="cuda"):
    """
    1) 从controller采样一个动作(即一套组合超参)
    2) 构建+训练+测试模型,得到acc
    3) 计算reward并更新controller
    """

    # ---------- 采样actions ----------
    state = torch.tensor([[0.0]], dtype=torch.float).to(device)
    actions, log_prob = controller.sample_actions(state)
    action_dict = controller.parse_actions(actions)
    print(f"[RL] Sampled actions: {action_dict}")

    # ---------- 数据准备 ----------
    train_data, val_data, test_data = load_dataset(dataset_name)
    # 对train/val/test每个graph做特征增强(示例是空壳)
    for dset in [train_data, val_data, test_data]:
        for data in dset:
            apply_feature_augment(data, action_dict["augment"])

    # 建议多线程
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    in_channels  = train_data.num_node_features
    out_channels = train_data.num_classes

    # ---------- 构建模型 ----------
    composed_model = build_model(action_dict, in_channels, out_channels).to(device)

    # ---------- 用Lightning封装并训练 ----------
    module = Evaluator(composed_model, strategy_name="RL-chosen", lr=action_dict["lr"])
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5
    )
    trainer.fit(module, train_loader, val_loader)
    result = trainer.test(module, test_loader)
    test_acc = result[0]["test_acc"]
    print(f"[RL] test_acc = {test_acc:.4f}")

    # 计算reward(可加复杂度惩罚), 这里就直接=acc
    reward = compute_reward(accuracy=test_acc, model_complexity=None, alpha=0.0)
    return log_prob, reward


def train_rl_controller():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controller = MultiHeadRLController(
        encoder_opts, pooling_opts, readout_opts, augment_opts,
        hidden_dim_opts, dropout_opts, lr_opts, bn_opts, temp_opts
    ).to(device)

    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)

    episodes = 5
    dataset_name = "PROTEINS"  # or "ENZYMES"
    for epi in range(episodes):
        print(f"========== RL Episode {epi} ==========")
        log_prob, reward = run_episode(dataset_name, controller, device=device)

        loss = controller.reinforce_loss(log_prob, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[RL] Episode {epi} => Reward={reward:.4f}, Loss={loss.item():.4f}")

    print("[RL] Training finished. You can now sample best actions or evaluate further.")


if __name__ == "__main__":
    train_rl_controller()
