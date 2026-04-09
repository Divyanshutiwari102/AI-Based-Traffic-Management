from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

PRESSURE_DENSITY_WEIGHT = 0.5
PRESSURE_QUEUE_WEIGHT = 0.3
PRESSURE_FORECAST_WEIGHT = 0.2
DEFAULT_CYCLE_BUDGET = 120
DEFAULT_MIN_GREEN = 10


@dataclass
class PPOProposal:
    green_times: list[int]


class PPOAgent:
    """Lightweight placeholder policy interface for future PPO integration."""

    def propose(self, density: Sequence[float], queue: Sequence[float] | None = None, forecast: Sequence[float] | None = None) -> PPOProposal:
        density = np.asarray(density, dtype=float)
        queue_vec = np.asarray(queue if queue is not None else np.zeros_like(density), dtype=float)
        forecast_vec = np.asarray(forecast if forecast is not None else density, dtype=float)

        pressure = (
            density * PRESSURE_DENSITY_WEIGHT
            + queue_vec * PRESSURE_QUEUE_WEIGHT
            + forecast_vec * PRESSURE_FORECAST_WEIGHT
        )
        pressure = np.clip(pressure, 0.01, None)
        ratios = pressure / np.sum(pressure)

        green = np.maximum((ratios * DEFAULT_CYCLE_BUDGET).astype(int), DEFAULT_MIN_GREEN)
        return PPOProposal(green_times=green.tolist())
