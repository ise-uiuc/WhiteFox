
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * 0.1
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
