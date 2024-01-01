
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 64, 1, stride=1)
    def forward(self, x6):
        v1 = self.conv(x6)
        v2 = v1 - 34.6097
        return v2
# Inputs to the model
x6 = torch.randn(1, 2, 38, 61)
