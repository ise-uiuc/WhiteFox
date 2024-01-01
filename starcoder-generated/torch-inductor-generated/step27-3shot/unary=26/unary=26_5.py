
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 2, 1)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.0950505
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(16, 2, 16, 16)
