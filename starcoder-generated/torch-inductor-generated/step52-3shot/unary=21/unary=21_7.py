
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 1)
        self.conv_2 = torch.nn.Conv2d(8, 8, 1)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x2)
        x1 = torch.tanh(x3)
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
