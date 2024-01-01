
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(128, 128, 3)
        self.conv_2 = torch.nn.Conv2d(128, 1, 1)
        self.conv_3 = torch.nn.Conv2d(128, 128, 3)
        self.conv_4 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv_5 = torch.nn.Conv2d(128, 128, 3)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x2 = torch.tanh(x2)
        x3 = self.conv_2(x2)
        x4 = self.conv_3(x3)
        x5 = self.conv_4(x4)
        x6 = self.conv_5(x4)
        x4 = torch.tanh(x5 + x6)
        x4 = x4 + x1
        return x4
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
