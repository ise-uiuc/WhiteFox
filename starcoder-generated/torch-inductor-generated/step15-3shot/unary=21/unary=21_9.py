
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, 1, padding=[1, 1], dilation=1, groups=1)
    def forward(self, x):
        x = self.conv(x)
        v2 = torch.tanh(x)
        return v2
# Inputs to the model
x = torch.randn(1, 16, 2, 2)
