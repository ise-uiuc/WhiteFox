
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.65, max_value=-3.43):
        super().__init__()
        self.prelu = torch.nn.PReLU(1, inplace=False)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.prelu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
