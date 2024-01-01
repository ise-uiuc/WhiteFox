
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.maxPool2d = torch.nn.MaxPool2d(3, stride=1, padding=0)
        self.conv = torch.nn.Conv1d(3, 18, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(18, 64, 3, stride=1, padding=1)
        self.avgPool = torch.nn.AvgPool2d(3, stride=2, padding=0)
    def forward(self, x1):
        v0 = self.relu(x1)
        v1 = self.maxPool2d(v0)
        v2 = self.conv(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv2(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.avgPool(v13)
        return v14
# Inputs to the model
x1 = torch.randn(2, 3, 1, 22, 11)
