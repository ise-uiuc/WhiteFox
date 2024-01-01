
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(83, 57, 9, stride=1, padding=0, output_padding=0, bias=False)
    def forward(self, x7):
        x1 = self.conv_t(x7)
        x2 = x1 > 0
        x3 = x1 * 0.167
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.interpolate(x4, scale_factor=(0.361, 0.266), recompute_scale_factor=None, mode='bilinear', align_corners=True)
# Inputs to the model
x7 = torch.randn(5, 83, 14, 17)
