
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, groups=1, dilation=1, padding=0, output_padding=0)
        self.bn = torch.nn.BatchNorm2d(8, eps=9.999999747378752e-06, momentum=0.800000011920929, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.bn(v1)
        v3 = v2 > 0
        v4 = v2 * 0.10000000149011612
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
