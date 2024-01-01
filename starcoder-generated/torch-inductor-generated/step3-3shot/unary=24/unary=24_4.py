
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        m2 = v1 > 0
        m3 = v1 * 0.01
        v4 = torch.where(m2, v1, m3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
