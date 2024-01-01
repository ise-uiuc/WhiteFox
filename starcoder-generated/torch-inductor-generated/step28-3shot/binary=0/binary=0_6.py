
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
    def forward(self, x1, x2, v2, v3, other=None):
        v1 = self.conv(x1)
        v4 = v1 + 1
        v5 = v4 + v2
        v6 = v5.mean()
        v7 = v6 + 1
        v8 = v7 + v3
        final = v8 + x2
        return final
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = 1
v2 = 1
v3 = 1
