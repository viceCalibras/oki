import torch.nn as nn

class MLPReg(nn.Module):
    def __init__(self):
        """Defines a simple MLP for regression task.
        """
        super(MLPReg, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(8, 16),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(32, 64),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.02),
            # nn.Dropout(p=0.2),
            nn.Linear(64, 2),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.model(x)
