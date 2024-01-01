
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 12, 7, stride=3, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 128, 128)
