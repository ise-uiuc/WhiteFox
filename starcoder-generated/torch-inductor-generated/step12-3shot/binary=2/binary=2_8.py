
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_weight = torch.nn.Parameter(torch.randn(8, 3))
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.conv2d(x1, self.conv_weight, bias=None, stride=1, padding=1, dilation=1, groups=1)
        v2 = v1 - torch.exp(torch.arange(8.0, 16.0))
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
