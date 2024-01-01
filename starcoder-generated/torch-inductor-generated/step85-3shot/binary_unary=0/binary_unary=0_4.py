
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(32, 64, 3)
        self.linear1 = torch.nn.Linear(1568, 1568)
        self.conv2d2 = torch.nn.Conv2d(64, 32, 3)
        self.linear2 = torch.nn.Linear(1568, 1568)
    def forward(self, x1, x2):
        v1 = self.conv2d1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = torch.flatten(v3, 1)
        v5 = self.linear1(v4)
        v6 = torch.tanh(v5)
        v7 = v6 + x2
        v8 = torch.relu(v7)
        v9 = self.conv2d2(v8)
        v10 = v9 + v8
        v11 = torch.relu(v10)
        v12 = torch.flatten(v11, 1)
        v13 = self.linear2(v12)
        v14 = torch.sigmoid(v13)
        return v14

# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
x2 = torch.randn(1, 1568)
