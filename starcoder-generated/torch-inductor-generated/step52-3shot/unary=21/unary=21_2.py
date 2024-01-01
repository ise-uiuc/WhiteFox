
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1)
        self.conv3 = torch.nn.Conv2d(8, 1, 1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(torch.tanh(x2))
        x4 = self.conv3(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
