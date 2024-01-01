
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 4, 2, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 2, 2, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 16, 10, 10)
