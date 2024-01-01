
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        t1 = v2 + 3
        v3 = torch.clamp(t1, 0, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
