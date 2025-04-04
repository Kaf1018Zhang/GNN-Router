def compute_reward(accuracy, model_complexity=None, alpha=0.0):
    """
    计算 RL controller 对应的奖励值。

    accuracy: 测试/验证集准确率 [0,1]
    model_complexity: 模型复杂度(可选，如FLOPs或参数量)
    alpha: 复杂度惩罚的权重

    返回: reward (float)
    """
    if model_complexity is not None:
        reward = accuracy - alpha * model_complexity
    else:
        reward = accuracy
    return reward
