
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, padding=1, dilation=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=2)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = torch.tanh(x2)
        x4 = self.conv2(x3)
        x5 = torch.tanh(x4)
        x6 = self.conv3(x5)
        return torch.tanh(x6)
# Inputs to the model
x = torch.randn(1, 32, 1, 1)
