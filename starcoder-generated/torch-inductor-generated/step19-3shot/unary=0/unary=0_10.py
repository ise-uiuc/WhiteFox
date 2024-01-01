
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 5, 3, stride=3, padding=2)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        x3 = torch.cat([x1, x2], 1)
        x4 = self.conv(x3)
        x5 = x4 * 0.5
        x6 = x4 * x4
        x7 = x6 * x4
        x8 = x7 * 0.044715
        x9 = x5 + x7
        x10 = x9 * 0.7978845608028654
        x11 = torch.tanh(x8)
        v10 = x11 + 1
        x13 = x10 * v10
        return x13
# Inputs to the model
x1 = torch.randn(1, 10, 384, 46)
x2 = torch.randn(1, 1, 384, 46)
