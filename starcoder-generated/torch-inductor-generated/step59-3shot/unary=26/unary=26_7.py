
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Conv2d(3, 93, 3, padding=1, stride=2, bias=True)
        self.conv_t = torch.nn.ConvTranspose1d(93, 66, 3, padding=1, stride=1, bias=False)
    def forward(self, x0):
        x1 = x0 + 5
        x2 = x1 + 1.3
        x3 = torch.nn.functional.max_pool2d(x2, 2, 2)
        x4 = x3 + 1
        x5 = x4 + 0.3
        x6 = x5 + 0.3
        x7 = x6 + 0.3
        x8 = x7 + 2.4
        x9 = x8 + 1.6
        x10 = self.t(x9)
        x11 = x10 + 1
        x12 = x11 + 1
        x13 = x12 * 0.8
        x14 = x13 - 0.6
        x15 = x14 + 0.9
        x16 = x15 * 0.6
        x17 = x16 - 1.6
        x18 = x17 + 1.1
        x19 = torch.nn.functional.dropout(x18, p=0.72)
        x20 = torch.add(1, 0)
        x21 = x7 + x20
        x22 = x21 * 0.8
        x23 = x22 - 1.4
        x24 = x23 + 1.3
        x25 = x24 * 0.7
        x26 = x25 - 1.7
        x27 = x26 * 1.4
        x28 = x27 - 1.4
        x29 = x19 * x28
        x30 = x29 * 0.6
        x31 = x30 * -0.7
        x32 = self.conv_t(32.0)
        x33 = x32 > 0
        x34 = x32 * self.negative_slope
        x35 = torch.where(x33, x32, x34)
        return torch.nn.functional.unfold(25.0, (2, 2)).reshape(-1)
# Inputs to the model
x0 = torch.randn(2, 3, 20, 25)
