
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 7, 3, padding=1, dilation=2, groups=1)
        self.conv2 = torch.nn.Conv2d(7, 14, 3, padding=1, dilation=2, groups=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return torch.tanh(v3)
# Inputs to the model
x = torch.randn(10, 1, 5, 5)
