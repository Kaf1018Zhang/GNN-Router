import random

class SimpleRoutingController:
    def __init__(self, candidates):
        """
        candidates: List[str]，候选 GNN 模块名称，如 ["GCN", "GAT", "GIN"]
        """
        self.candidates = candidates
        self.best_choice = {}

    def sample_random(self):
        """随机采样一个模块名称。"""
        return random.choice(self.candidates)

    def select_module(self, dataset_name):
        """
        若已设置best_choice，则使用；否则默认返回 candidates[0]。
        """
        return self.best_choice.get(dataset_name, self.candidates[0])

    def set_best(self, dataset_name, module_name):
        """记录在 dataset_name 上的最佳策略。"""
        self.best_choice[dataset_name] = module_name
