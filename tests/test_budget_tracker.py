import pytest
from privrag.core.budget_tracker import PrivacyBudgetTracker
def test_budget_consumption():
    tracker = PrivacyBudgetTracker(max_epsilon=5.0)
    assert tracker.consume(2.0) is True
    assert tracker.remaining_epsilon == pytest.approx(3.0)
    assert tracker.consume(4.0) is False
