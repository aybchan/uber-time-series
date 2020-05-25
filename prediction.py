import torch
import torch.nn as nn

class Prediction(nn.Module):
    def __init__(self, n_in):
        super(Prediction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x