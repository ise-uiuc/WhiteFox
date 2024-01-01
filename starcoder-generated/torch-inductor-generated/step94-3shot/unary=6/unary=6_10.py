
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 20, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(20)
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(800, 120)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, 10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.flatten(v2)
        v4 = self.linear1(v3)
        v5 = self.relu(v4)
        v6 = self.linear2(v5)
        v7 = self.linear3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 16, 28, 28)
