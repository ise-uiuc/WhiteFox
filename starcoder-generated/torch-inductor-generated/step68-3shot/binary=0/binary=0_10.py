
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 1, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x1, x2, x3, x4):
        x5 = self.conv(x1)
        x6 = x5 + 1.0
        x7 = x6 + x2
        x8 = x7 + x3
        x9 = x8 + x4
        return x9
# Inputs to the model
x1 = torch.randn(1, 3, 64)
x2 = 1
x3 = True
x4 = torch.zeros(1, 3, 64)
