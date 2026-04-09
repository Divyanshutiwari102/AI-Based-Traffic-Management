from __future__ import annotations

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden: int = 64, layers: int = 2, horizon: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PredictorService:
    def __init__(self):
        self.model = TrafficLSTM()
        self.model.eval()

    def predict(self, sequence_tensor):
        with torch.no_grad():
            return self.model(sequence_tensor)
