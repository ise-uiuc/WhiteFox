
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 5, stride=1, padding=0, output_padding=0, groups=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, eps=0.03564, momentum=0.011778)
    def forward(self, x1):
        v1 = self.conv_transpose(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        v5 = self.bn(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
