
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(1, 3, 1, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(6, 5, 7, stride=2, padding=1)
    def forward(self, x3):
        v1 = self.conv1d(x3)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv(v10)
        return v11
# Inputs to the model
x3 = torch.randn(1, 1, 128, 14)
