
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1.tanh().clamp_min(0) + 1
        v3 = v2 / 2
        v4 = -v3 + 2
        v5 = self.conv_2(v4)
        v6 = self.conv_3(v5).clamp_min(0)
        return v6
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
