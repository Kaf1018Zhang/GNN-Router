import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.routing_controller import SimpleRoutingController
from experiments.deploy_controller import run_with_fixed_strategy
from core.reward_utils import compute_reward

def train_controller_with_reward(dataset_name, candidates, trials=5):
    """
    多次随机采样不同 GNN 模块组合，
    计算其在验证集/测试集上的准确率并作为 reward，
    从而选出最佳策略并保存。
    """
    controller = SimpleRoutingController(candidates)
    results = {}

    for _ in range(trials):
        choice = controller.sample_random()
        print(f"[Controller Trainer] Trying strategy: {choice}")

        # 利用该策略跑一次完整训练+验证+测试，得到 test_acc
        acc = run_with_fixed_strategy(dataset_name, controller, override_choice=choice)

        # 此处可加复杂度(如 params)指标
        complexity = 0.0
        reward = compute_reward(accuracy=acc, model_complexity=complexity, alpha=0.0)
        results[choice] = max(reward, results.get(choice, 0))

    # 找到分数最高的策略
    best = max(results, key=results.get)
    controller.set_best(dataset_name, best)
    print(f"[Controller Trainer] Best strategy for {dataset_name}: {best}")
    return controller

if __name__ == "__main__":
    candidates = ["GCN", "GAT", "GIN"]

    print("=== Training strategy controller on PROTEINS ===")
    controller_proteins = train_controller_with_reward("PROTEINS", candidates)

    print("=== Training strategy controller on ENZYMES ===")
    controller_enzymes = train_controller_with_reward("ENZYMES", candidates)
