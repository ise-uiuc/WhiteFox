
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(30, 26, 7, stride=1, padding=0, bias=False, groups=18)
    def forward(self, x1):
        a1 = self.conv_t(x1)
        a2 = a1 > 0
        a3 = a1 * 0.668
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x1 = torch.randn(16, 30, 24, 96, device='cpu')
