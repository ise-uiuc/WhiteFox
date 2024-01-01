
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 4, stride=2, padding=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.3
        x4 = torch.where(x2, x1, x3)
        x5 = x4 * 1.45
        x6 = x5 + 0.5
        return torch.round(x6)
# Inputs to the model
x = torch.randn(1, 4, 22, 37)
