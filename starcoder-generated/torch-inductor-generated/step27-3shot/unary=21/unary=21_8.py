
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 1, 1, dilation=2, padding=2)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, dilation=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 4, 3, 3)
