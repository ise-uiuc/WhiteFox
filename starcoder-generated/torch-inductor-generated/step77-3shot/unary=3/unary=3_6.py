
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d1 = torch.nn.Conv1d(1, 1, 1, stride=1, padding=0)
        self.conv1d2 = torch.nn.ConvTranspose1d(1, 1, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1d1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1d2(v6)
        v8 = v7 * 0.5
        v9 = self.relu(v8)
        v10 = v1 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v7 * v12
        return v9-v13
# Inputs to the model
x1 = torch.randn(3, 1, 20)
