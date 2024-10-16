import torch.nn as nn

class MLPRegSimulation(nn.Module):
    def __init__(self):
        """Defines a simple MLP for regression task.
        """
        super(MLPRegSimulation, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(8, 16),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(32, 64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.model(x)

class MLPRegRoughness(nn.Module):
    def __init__(self):
        """Defines a simple MLP for regression task - Roughness dataset.
        """
        super(MLPRegRoughness, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)