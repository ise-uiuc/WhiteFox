
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, dilation=2)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 16
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
