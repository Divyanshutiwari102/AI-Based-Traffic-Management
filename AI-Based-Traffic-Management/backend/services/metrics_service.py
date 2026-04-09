from __future__ import annotations

import numpy as np


def compute_jain_fairness(values):
    arr = np.asarray(values, dtype=float)
    numerator = np.sum(arr) ** 2
    denominator = len(arr) * np.sum(arr ** 2) + 1e-9
    return float(numerator / denominator)


def compute_reward(waiting_times, queue_lengths):
    awt = float(np.mean(waiting_times)) if waiting_times else 0.0
    max_q = float(np.max(queue_lengths)) if queue_lengths else 0.0
    jfi = compute_jain_fairness(waiting_times if waiting_times else [0.0, 0.0, 0.0, 0.0])
    return -0.6 * awt - 0.3 * max_q + 0.1 * jfi
