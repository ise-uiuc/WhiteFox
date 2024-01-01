
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 16, 2, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(16, 64, 2, stride=1, padding=0)
        self.conv_3 = torch.nn.Conv2d(64, 256, 2, stride=1, padding=0)
        self.conv_4 = torch.nn.Conv2d(256, 128, 2, stride=1, padding=0)
        self.conv_5 = torch.nn.Conv2d(128, 1, 2, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = torch.tanh(x1)
        x3 = self.conv_2(x)
        x4 = torch.tanh(x3)
        x5 = self.conv_3(x)
        x6 = torch.tanh(x5)
        x7 = self.conv_4(x)
        x8 = torch.tanh(x7)
        x9 = self.conv_5(x)
        x10 = torch.tanh(x9)
        return x10
# Inputs to the model
x = torch.randn(1, 1, 500, 500)
