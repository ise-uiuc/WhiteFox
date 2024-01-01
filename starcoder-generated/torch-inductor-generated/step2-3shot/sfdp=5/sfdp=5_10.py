
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
  ...


# Initializing the model
query = torch.randn(1, 1, 1, 1)
key = torch.randn(1, 1, 1, 1)
value = torch.randn(1, 1, 1, 1)
dropout_p = 0.0
mask = None
attn_weight, output = attention(query, key, value, dropout_p, mask)

from typing import List, Tuple

import torch
from torch.optim import Optimizer


def train_loop(optimizer: Optimizer, model: torch.nn.Module) -> None:
    # Iterate through a dataset

    for x, y in train_dataset:
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        optimizer.step()


# Initializing optimizer and model
model = MnistModel(10)
optimizer = Adam(lr=0.001, params=model.parameters())

train_loop(optimizer, model)

