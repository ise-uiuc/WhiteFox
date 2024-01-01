
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x8):
        v1 = self.conv(x8)
        v2 = v1 + torch.tensor([0.0, 1.0])
        return v2
# Inputs to the model
x8 = torch.randn(1, 1, 8, 8)
