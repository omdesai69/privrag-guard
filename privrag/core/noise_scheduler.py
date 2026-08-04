def compute_laplace_scale(sensitivity: float, epsilon: float) -> float:
    if epsilon <= 0: raise ValueError("Epsilon must be positive")
    return sensitivity / epsilon
