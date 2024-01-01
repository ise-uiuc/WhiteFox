
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(40, 16, 3, stride=[2, 1, 1], padding=[1, 2, 2], output_padding=[1, 1, 2], dilation=1, groups=1, bias=True)
    def forward(self, x2):
        z1 = torch.cat((x2[0:1, :, :, :, :], x2[:, :, 0:1, :, :], x2[:, :, :, 0:1, :], x2[:, :, :, :, 0:1, :]), 1)
        z2 = torch.cat((x2[-1:, :, :, :, :].flip([0]), x2[:, :, -1:, :, :].flip([1]), x2[:, :, :, -1:, :].flip([2]), x2[:, :, :, :, -1:, :].flip([3])), 1)
        z3 = self.conv_t(z1)
        z4 = z3 > 0
        z5 = z3 * -0.00016357
        z6 = torch.where(z4, z3, z5)
        x = torch.cat((z6, z2[1:-1, :, :, :, :], z3[0:1, :, :, :, :], z3[:, :, 0:1, :, :], z3[:, :, :, 0:1, :], z3[:, :, :, :, 0:1, :]), 1)
        y = torch.cat((z1[:, -1:, :, :, :].flip([1]), x[:, :, -1:, :, :].flip([2]), x[:, :, :, -1:, :].flip([3])), 1)
        return torch.nn.functional.interpolate(y, size=[22, 2, 2, 2, 2], mode='trilinear', align_corners=None, recompute_scale_factor=None)
# Inputs to the model
x2 = torch.randn(2, 40, 3, 23, 27)
