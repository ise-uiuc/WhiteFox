
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x2)
        x4 = x3 + 6
        x5 = torch.relu6(x4)
        x6 = x2 - 6
        x7 = torch.clamp_max(x6, 6)
        x8 = x7 + x2
        x9 = x5 * x8
        x10 = x9 * 0.5
        return x10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
