"""Differential privacy budget composition tracker."""
from dataclasses import dataclass
from typing import List

@dataclass
class PrivacyQuery:
    epsilon: float
    delta: float

class PrivacyBudgetTracker:
    def __init__(self, max_epsilon: float = 10.0, max_delta: float = 1e-4):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.history: List[PrivacyQuery] = []
    def consume(self, epsilon: float, delta: float = 0.0) -> bool:
        if sum(q.epsilon for q in self.history) + epsilon > self.max_epsilon:
            return False
        self.history.append(PrivacyQuery(epsilon=epsilon, delta=delta))
        return True
    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.max_epsilon - sum(q.epsilon for q in self.history))
