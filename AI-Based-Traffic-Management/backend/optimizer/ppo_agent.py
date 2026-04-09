from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PPOProposal:
    green_times: list[int]


class PPOAgent:
    """Lightweight placeholder policy interface for future PPO integration."""

    def propose(self, density: Sequence[float], queue: Sequence[float] | None = None, forecast: Sequence[float] | None = None) -> PPOProposal:
        density = np.asarray(density, dtype=float)
        queue_vec = np.asarray(queue if queue is not None else np.zeros_like(density), dtype=float)
        forecast_vec = np.asarray(forecast if forecast is not None else density, dtype=float)

        pressure = density * 0.5 + queue_vec * 0.3 + forecast_vec * 0.2
        pressure = np.clip(pressure, 0.01, None)
        ratios = pressure / np.sum(pressure)

        cycle_budget = 120
        green = np.maximum((ratios * cycle_budget).astype(int), 10)
        return PPOProposal(green_times=green.tolist())
