import torch.nn as nn
import torch.nn.functional as F
import torch

class SCENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4419, 2210, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(2210)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(2210, 4419, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(4419)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4419, 2)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x += shortcut
        x = self.relu2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x