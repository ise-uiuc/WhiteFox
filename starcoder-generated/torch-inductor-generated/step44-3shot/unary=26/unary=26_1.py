
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 7, 3, stride=1, padding=0, bias=False, output_padding=0)
        self.negative_slope = -0.25
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x3 = torch.randn(5, 8, 32, 14, device='cuda')
