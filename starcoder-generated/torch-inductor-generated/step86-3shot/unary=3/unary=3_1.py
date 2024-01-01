
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 7, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(7, 11, 5, stride=2, padding=5)
    def forward(self, x1):
        v1 = x1.data
        v2 = v1 * 0.70710678
        v3 = v2 + 0.5
        v4 = self.conv(v3)
        v5 = v4 * 0.12940952
        v6 = v4 + 0.99438503
        v7 = v5 * v6
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 20, 20)
