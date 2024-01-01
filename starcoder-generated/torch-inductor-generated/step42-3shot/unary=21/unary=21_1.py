
import torch.nn as nn
class ModelTanh(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 200, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, bias=False)
        self.conv3 = torch.nn.Conv3d(16, 32, 3, stride=3, bias=False)
        self.conv4 = torch.nn.Conv2d(16, 8, kernel_size=2, stride=(2, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 42, 287)
