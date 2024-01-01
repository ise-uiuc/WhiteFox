
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 2, stride=2, padding=7, dilation=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.99
        return v2
# Inputs to the model
x = torch.randn(4, 2, 8, 8)
