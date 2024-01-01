
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 5, (19, 5), stride=(19, 5), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(5, 2, (9, 2), stride=(9, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 6, 12)
