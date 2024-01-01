
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(33, 23, 3, stride=7, padding=1)
        self.negative_slope = 0.0625
    def forward(self, x2):
        x1 = self.conv_t(x2)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x2 = torch.randn(7, 33, 20, 25, device='cuda')
