
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 5, stride=1, padding=2, dilation=2, groups=1, bias=False, padding_mode='zeros')
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
