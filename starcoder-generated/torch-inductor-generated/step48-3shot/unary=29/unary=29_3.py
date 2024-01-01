
class Model(torch.nn.Module):
    def __init__(self, min_value=-8.62, max_value=-8.74):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, 2, stride=2, groups=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, eps=1.64, momentum=0.4)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 9, 7)
