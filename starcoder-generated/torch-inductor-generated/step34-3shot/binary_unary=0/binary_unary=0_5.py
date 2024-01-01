
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1)
        self.linear1 = torch.nn.Linear(64*64, 4)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1)
        self.linear2 = torch.nn.Linear(64*64, 4)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv(x1)
        v2 = v1.view(-1, 64*64)
        v3 = self.linear1(v2)
        v4 = v1 + x2
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6.view(-1, 64*64)
        v8 = self.linear2(v7)
        v9 = v8 + x3
        v10 = torch.relu(v9)
        v11 = v10 + x4
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 4)
x4 = torch.randn(1, 4)
