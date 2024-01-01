
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(2, 32, 3)
        self.conv2d = nn.Conv2d(1, 32, 3)
        self.linear = nn.Linear(4, 1)

    def forward(self, x1, x_2d):
        output1_1d = self.conv1d(x1)
        output2_1d = output1_1d - -0.0
        output3_1d = F.relu(output2_1d)

        output1_2d = self.conv2d(x_2d)
        output2_2d = output1_2d - 50
        output3_2d = F.relu(output2_2d)

        output = torch.cat([output3_1d, output3_2d], dim=1)
        output = self.linear(output)

        return output
# Inputs to the model
x1 = torch.randn(100, 2, 300)
x2 = torch.randn(1, 1, 48, 48)
