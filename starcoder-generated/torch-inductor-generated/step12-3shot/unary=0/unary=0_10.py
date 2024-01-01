
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 5, stride=1, padding=0, dilation=1)
        self.conv1 = torch.nn.Conv2d(12, 4, 3, stride=1, padding=0, dilation=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = self.conv1(v1)
        v3 = v2 + 1
        v4 = v1 * v3
        return v4
# Inputs to the model
x2 = torch.randn(1, 1, 80, 80)
