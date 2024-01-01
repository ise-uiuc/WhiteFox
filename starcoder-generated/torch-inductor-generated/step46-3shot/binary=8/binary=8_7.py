
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        m1 = x1 + x2
        m2 = x2 + x3
        m3 = x1 + x3
        m4 = m1 + m2 + m3
        v1 = self.conv(m4)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
x2 = torch.randn(1, 1, 16, 16)
x3 = torch.randn(1, 1, 16, 16)
