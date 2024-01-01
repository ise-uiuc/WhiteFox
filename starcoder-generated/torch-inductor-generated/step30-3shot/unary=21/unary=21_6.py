
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 32, 1)
        self.conv_2 = torch.nn.Conv2d(32, 32, 1, groups=2)
        self.conv_3 = torch.nn.Conv2d(32, 1, 1, dtype=torch.float)
    def forward(self, x1):
        x2 = self.conv_1(x1).to(torch.float)
        x3 = self.conv_2(x2)
        x1 = torch.nn.Tanh()(x3)
        x1 = self.conv_3(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
