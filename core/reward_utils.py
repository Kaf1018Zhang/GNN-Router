def compute_reward(accuracy, model_complexity=None, alpha=0.0):
    """
    Computes a reward for RL controller.

    Parameters:
    - accuracy (float): validation accuracy [0,1]
    - model_complexity (float): optional complexity cost (e.g., FLOPs, params)
    - alpha (float): weight for complexity penalty

    Returns:
    - reward (float): weighted reward
    """
    if model_complexity is not None:
        reward = accuracy - alpha * model_complexity
    else:
        reward = accuracy
    return reward