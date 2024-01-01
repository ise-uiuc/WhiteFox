
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=2, padding=2, dilation=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 7.32
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
