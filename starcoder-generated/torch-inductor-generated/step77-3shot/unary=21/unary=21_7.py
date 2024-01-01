
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(6, 8, 5)
        self.conv_2 = torch.nn.Conv2d(8, 4, 5)
        self.conv_3 = torch.nn.Conv2d(4, 6, 3)
        self.conv_4 = torch.nn.Conv2d(6, 2, 3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = torch.tanh(x2)
        x4 = self.conv_2(x3)
        x5 = torch.tanh(x4)
        x6 = self.conv_3(x5)
        x7 = torch.tanh(x6)
        x8 = self.conv_4(x7)
        return x8
# Inputs to the model
x = torch.randn(1, 6, 10, 10)
