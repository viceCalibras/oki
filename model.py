import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 9),
    nn.ReLU(),
    nn.Linear(9, 18),
    nn.ReLU(),
    nn.Linear(18, 9),
    nn.ReLU(),
    nn.Linear(9, 2),
)
