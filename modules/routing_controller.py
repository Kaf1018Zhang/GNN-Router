import random

class SimpleRoutingController:
    def __init__(self, candidates):
        self.candidates = candidates
        self.best_choice = {name: random.choice(candidates) for name in ["PROTEINS", "ENZYMES"]}

    def select_module(self, dataset_name):
        return self.best_choice[dataset_name]

    def train_on_dataset(self, dataset_name):
        print(f"[RoutingController] Searching best module for {dataset_name}...")
        # 🔧 placeholder: randomly select (replace with RL later)
        self.best_choice[dataset_name] = random.choice(self.candidates)
        print(f"[RoutingController] Selected: {self.best_choice[dataset_name]}")
