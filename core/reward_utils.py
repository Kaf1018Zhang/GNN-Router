def compute_reward(accuracy, model_complexity=None, alpha=0.0):
    if model_complexity is not None:
        reward = accuracy - alpha * model_complexity
    else:
        reward = accuracy
    return reward
