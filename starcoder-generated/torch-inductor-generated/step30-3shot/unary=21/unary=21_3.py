
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(16, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x2)
        x4 = self.conv_3(x3)
        x5 = torch.tanh(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
