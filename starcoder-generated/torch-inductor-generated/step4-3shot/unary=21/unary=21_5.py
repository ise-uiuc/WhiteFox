
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 16, 1)
        self.conv2 = torch.nn.Conv2d(16, 1, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.tanh(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 2, 49, 49)
