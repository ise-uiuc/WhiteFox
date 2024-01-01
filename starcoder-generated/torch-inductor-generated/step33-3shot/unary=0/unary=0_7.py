
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 32, 5, stride=1, padding=2, dilation=1, groups=1, bias=False)
        self.conv2 = torch.nn.Conv3d(32, 64, 3, stride=1, padding=1, dilation=2, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
x1 = torch.randn(12, 1, 40, 115, 136)
