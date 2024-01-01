
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = torch.nn.Conv2d(3, 2, 3, groups=2, padding=1)
        self.conv_dw3x3 = torch.nn.Conv2d(2, 2, 3, groups=2, stride=2, padding=1)
        self.conv_dw5x5 = torch.nn.Conv2d(2, 2, 5, groups=2, stride=2, padding=2)
    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv_dw3x3(x1)
        x3 = self.conv_dw5x5(x1)
        return x2
# Inputs (x7) and (x8) to the model
x7 = torch.randn(1, 3, 256, 256)
x8 = torch.randn(1, 3, 128, 128)
