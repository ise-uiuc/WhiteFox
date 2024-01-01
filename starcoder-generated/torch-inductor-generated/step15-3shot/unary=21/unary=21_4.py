
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, 1, 2, groups=1, bias=False, dilation=1)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x4 = torch.randn(64, 1, 3, 3)
