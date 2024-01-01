
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(64, 160, 1, stride=1, padding=0, dilation=1, groups=1)
        self.conv1 = torch.nn.Conv1d(160, 96, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x2, x1):
        v1 = self.conv(x2)
        v2 = self.conv1(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v2 * v10
# Inputs to the model
x2 = torch.randn(1, 64, 256)
x1 = torch.randn(1, 256, 4)
