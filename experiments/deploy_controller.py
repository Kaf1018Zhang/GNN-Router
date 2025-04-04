import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import torch
from datasets.loader_factory import load_dataset
from modules.gnn_base import GCNBlock, GATBlock, GINBlock
from modules.pooling import global_mean_pooling
from modules.routing_controller import SimpleRoutingController
from core.module_executor import ModuleExecutor
from utils.evaluator import Evaluator


def run_with_fixed_strategy(dataset_name, controller, override_choice=None):
    """
    根据给定 controller 的策略，在指定 dataset 上进行单次训练和测试，
    返回 test_acc。
    """
    train_set, val_set, test_set = load_dataset(dataset_name)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    gnn_candidates = {
        "GCN": GCNBlock,
        "GAT": GATBlock,
        "GIN": GINBlock
    }

    # 若 override_choice 不为空，使用它；否则使用controller已有配置
    strategy = override_choice if override_choice else controller.select_module(dataset_name)
    print(f"[Deploy] Using strategy: {strategy} on dataset: {dataset_name}")

    # 获取输入和输出维度（TUDataset中 num_node_features 和 num_classes）
    in_channels = train_set.num_node_features
    out_channels = train_set.num_classes

    # 构建组合模型
    model = ModuleExecutor(
        gnn_block=gnn_candidates[strategy],
        pooling_fn=global_mean_pooling,
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        dropout=0.5
    )

    # LightningModule 封装
    evaluator_module = Evaluator(model, strategy_name=strategy, lr=0.001)

    # 训练器
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5
    )

    # 训练 + 验证
    trainer.fit(evaluator_module, train_loader, val_loader)
    # 测试
    result = trainer.test(evaluator_module, test_loader)
    test_acc = result[0]['test_acc']
    print(f"[Deploy] Final test accuracy on {dataset_name} = {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    # 示例：手动设置最优策略并部署
    controller = SimpleRoutingController(["GCN", "GAT", "GIN"])
    # 假设我们之前搜索的最佳策略是：PROTEINS -> "GCN"，ENZYMES -> "GAT"
    controller.set_best("PROTEINS", "GCN")
    controller.set_best("ENZYMES", "GAT")

    # 部署测试
    acc_proteins = run_with_fixed_strategy("PROTEINS", controller)
    acc_enzymes = run_with_fixed_strategy("ENZYMES", controller)
