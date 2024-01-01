
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(253, 79, 1, stride=1, padding=1, bias=False)
    def forward(self, x10):
        x11 = self.conv_t(x10)
        x12 = x11 > 0
        x13 = x11 * 0.148
        x14 = torch.where(x12, x11, x13)
        x15 = torch.nn.functional.adaptive_avg_pool2d(x14, (1, 1))
        x16 = torch.nn.functional.interpolate(x15, (24, 55), mode='nearest', align_corners=None)
        x17 = torch.nn.functional.interpolate(torch.nn.ReLU()(x16), size=(6, 40))
        return x17
# Inputs to the model
x10 = torch.randn(1, 253, 93, 45)
