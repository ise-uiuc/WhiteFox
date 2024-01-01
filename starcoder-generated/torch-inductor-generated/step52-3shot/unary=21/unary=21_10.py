
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 16, 1, stride=1)
        self.conv_2 = torch.nn.Conv2d(1, 32, 1, stride=1)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x2 = torch.tanh(x2)
        x2 = self.conv_2(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
