
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.relu(v5)
        return v6
# Inputs to the model
import torchvision
x = torch.randn(2, 32, 256, 256)
