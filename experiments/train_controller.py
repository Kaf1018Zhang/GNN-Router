from modules.routing_controller import SimpleRoutingController
from experiments.deploy_controller import run_with_fixed_strategy
from core.reward_utils import compute_reward


def train_controller_with_reward(dataset_name, candidates, trials=5):
    controller = SimpleRoutingController(candidates)
    results = {}

    for _ in range(trials):
        choice = controller.sample_random()
        print(f"[Controller Trainer] Trying strategy: {choice}")

        # Run single evaluation with given strategy
        acc = run_with_fixed_strategy(dataset_name, controller=controller, override_choice=choice)

        # Optional: include model complexity if available
        complexity = 0.0
        reward = compute_reward(accuracy=acc, model_complexity=complexity, alpha=0.0)
        results[choice] = max(reward, results.get(choice, 0))

    # Select best strategy
    best = max(results, key=results.get)
    controller.set_best(dataset_name, best)
    print(f"[Controller Trainer] Best strategy for {dataset_name}: {best}")
    return controller


if __name__ == "__main__":
    candidates = ["GCN", "GAT", "GIN"]
    print("Training strategy controller on PROTEINS...")
    controller = train_controller_with_reward("PROTEINS", candidates)

    print("Training strategy controller on ENZYMES...")
    controller = train_controller_with_reward("ENZYMES", candidates)
