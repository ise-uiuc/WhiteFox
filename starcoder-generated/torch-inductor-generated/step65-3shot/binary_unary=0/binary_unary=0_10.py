
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = x1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        return v4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        )
        self.output = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.input(x1)
        v2 = self.input(x2)
        v3 = v1
        if v1.sum() > v2.sum():
            v3 = v2
        v4 = v3 + v3
        v5 = torch.relu(v4)
        v6 = self.output(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
