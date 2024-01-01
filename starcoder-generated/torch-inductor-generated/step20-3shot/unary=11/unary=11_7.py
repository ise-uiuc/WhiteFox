
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 6, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(6, eps=0.447006, momentum=0.091515)
    def forward(self, x1):
        v1 = self.conv_transpose(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        v5 = self.bn(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
