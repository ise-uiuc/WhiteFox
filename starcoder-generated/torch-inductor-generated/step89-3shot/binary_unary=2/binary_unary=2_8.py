
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32*28*28, 28*28)
        self.fc2 = torch.nn.Linear(28*28, 28*28)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0
        v3 = v2.flatten(1)
        v4 = self.fc1(v3)
        v5 = v4 - False
        v6 = v5.flatten(1)
        v7 = self.fc2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(16, 3, 64, 64)
