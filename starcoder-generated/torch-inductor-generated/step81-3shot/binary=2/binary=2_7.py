
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(1, 5, 3, stride=1, padding=0)
        self.conv_b = torch.nn.Conv2d(4, 6, 7, stride=1, padding=0)
    def forward(self, x1, x2, x3, x4):
        v0 = self.conv_a(x1)
        v1 = np.full(shape=(0, ), fill_value=1.0)
        v2 = v1 - 2
        v3 = self.conv_b(v2)
        v4 = v3 + x4
        v5 = v4 - x1
        return v4
# Inputs for the model
x1 = torch.randn(1, 1, 10, 10)
x2 = torch.randn(1, 1, 12, 12)
x3 = torch.randn(1, 1, 14, 14)
x4 = torch.randn(1, 1, 16, 16)
