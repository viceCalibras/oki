import torch.nn as nn

class MLPReg(nn.Module):
    def __init__(self):
        """Defines a simple MLP for regression task.
        """
        super(MLPReg, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            # nn.Linear(8, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 32),
            # nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            # nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.model(x)
