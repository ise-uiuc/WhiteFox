
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(633, 272, 4, stride=1, padding=2, bias=False, dilation=1, groups=1, output_padding=0)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.436
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool2d(x4, (6, 8)) + x4 + x4
# Inputs to the model
x = torch.randn(78, 633, 17, 16)
