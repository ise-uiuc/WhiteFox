
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(2, 16, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(1572864, 1000)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv2(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = v3 + x3
        v5 = torch.relu(v4)
        v6 = self.linear(v5.reshape(1, -1))
        v7 = v6 + x4
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 224, 224)
x2 = torch.randn(1, 2, 224, 224)
x3 = torch.randn(1, 2, 224, 224)
x4 = torch.randn(1, 1000)
