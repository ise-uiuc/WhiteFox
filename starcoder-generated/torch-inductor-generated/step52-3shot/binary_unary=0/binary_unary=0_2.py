
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(16, 16)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.dense1(x1)
        v2 = v1[:, None, None] + x1
        v3 = self.conv1(v2)
        v4 = v3 + v2
        v5 = v4 + x2
        v6 = torch.nn.ReLU()(v5)
        v7 = self.conv2(v6)
        v8 = v7 + x3
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
