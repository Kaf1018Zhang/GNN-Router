from datasets.loader_factory import load_dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from modules.gnn_base import GCNBlock, GATBlock, GINBlock
from modules.pooling import global_mean_pooling
from core.module_executor import ModuleExecutor
from utils.evaluator import Evaluator
from modules.routing_controller import SimpleRoutingController


def run_with_fixed_strategy(dataset_name, controller, override_choice=None):
    train_set, val_set, test_set = load_dataset(dataset_name)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    gnn_candidates = {
        "GCN": GCNBlock,
        "GAT": GATBlock,
        "GIN": GINBlock,
    }

    strategy = override_choice if override_choice else controller.select_module(dataset_name)
    print(f"[Deploy] Using strategy: {strategy} on dataset: {dataset_name}")

    model = ModuleExecutor(
        gnn_block=gnn_candidates[strategy],
        pooling_fn=global_mean_pooling,
        in_channels=train_set.num_node_features,
        hidden_channels=64,
        out_channels=train_set.num_classes,
        dropout=0.5
    )

    module = Evaluator(model, strategy_name=strategy)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if pl.utilities.device_parser.num_cuda_devices() > 0 else "cpu",
        devices=1,
        log_every_n_steps=5
    )
    trainer.fit(module, train_loader, val_loader)
    result = trainer.test(module, dataloaders=test_loader)
    return result[0]['test_acc']  # for reward computation if needed


if __name__ == "__main__":
    controller = SimpleRoutingController(["GCN", "GAT", "GIN"])
    controller.set_best("PROTEINS", "GCN")
    controller.set_best("ENZYMES", "GAT")

    run_with_fixed_strategy("PROTEINS", controller)
    run_with_fixed_strategy("ENZYMES", controller)
