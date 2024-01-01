
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, (3, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2 + v3
        v5 = v4
        v6 = v5
        v7 = v6
        v8 = v7
        v9 = v8
        v10 = v9
        v11 = v10
        v12 = v11
        v13 = torch.tanh(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
