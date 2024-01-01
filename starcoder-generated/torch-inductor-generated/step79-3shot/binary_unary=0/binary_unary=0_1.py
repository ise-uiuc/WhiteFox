
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 + x1
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 + v1
        t1 = torch.sigmoid(v6)
        v7 = torch.relu(t1)
        v8 = 10 + v7
        t2 = torch.sigmoid(v8)
        v9 = torch.relu(t2)
        v10 = torch.mul(x3, v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
