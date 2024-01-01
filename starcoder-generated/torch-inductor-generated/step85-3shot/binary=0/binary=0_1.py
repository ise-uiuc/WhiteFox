
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, stride=2, padding=3, dilation=3)
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
