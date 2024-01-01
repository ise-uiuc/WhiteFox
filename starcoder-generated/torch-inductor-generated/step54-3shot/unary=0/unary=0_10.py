
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(2, 128, 1, stride=1, padding=0)
    def forward(self, x20):
        v4 = x20.view(2, 1, 1, 1, -1)
        v1 = self.conv(v4)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v7 = self.conv(v4)
        v5 = v3 * v7
        v6 = v5 * 0.044715
        v8 = self.conv(v4)
        v9 = v8 + v6
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v2 * v12
        return v13.view(1, -1)
# Inputs to the model
x20 = torch.randn(2, 2, 1, 1)
