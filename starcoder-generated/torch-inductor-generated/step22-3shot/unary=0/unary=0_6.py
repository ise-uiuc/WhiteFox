
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(91, 2, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(2)
        self.fc = torch.nn.Linear(2, 1284)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        v12 = self.fc(v11)
        v13 = v12 * 0.5
        v14 = v12 * v12
        v15 = v14 * v12
        v16 = v15 * 0.044715
        v17 = v12 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v13 * v20
        return v21
# Inputs to the model
x1 = torch.randn(1, 91, 256, 4)
