
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 32, 1)
        self.conv_2 = torch.nn.Conv2d(1, 16, 3, groups=8)
        self.conv_3 = torch.nn.Conv2d(32, 16, 1)
        self.conv_4 = torch.nn.Conv2d(16, 1, 7)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = torch.tanh(x2)
        x4 = self.conv_2(x3)
        x5 = self.conv_3(x4)
        x6 = torch.nn.Tanh()(x5)
        x7 = self.conv_4(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
