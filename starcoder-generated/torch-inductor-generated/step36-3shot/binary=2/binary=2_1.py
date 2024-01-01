
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=2, dilation=2)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 - True
        return v2
# Inputs to the model
x5 = torch.randn(1, 3, 64, 64)
