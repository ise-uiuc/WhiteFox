
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.toto1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.toto3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x3):
        v1 = self.toto1(x1)
        v2 = v1 + x3
        v3 = torch.tanh(v2)
        v4 = self.toto1(v3)
        v5 = v4 + x3
        v6 = torch.tanh(v5)
        v7 = self.toto3(v6)
        v8 = v7 + x3
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x3 = 2
