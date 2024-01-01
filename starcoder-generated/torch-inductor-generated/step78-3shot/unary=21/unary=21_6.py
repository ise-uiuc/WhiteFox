
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 32,  5, 1, 2, dilation=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 2, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = self.conv2(x2)
        x4 = torch.tanh(x3)
        x5 = self.conv3(x4)
        return x5
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
