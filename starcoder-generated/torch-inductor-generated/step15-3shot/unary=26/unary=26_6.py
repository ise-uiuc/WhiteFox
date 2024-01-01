
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
    def forward(self, x1):
        t1 = self.conv_t(x1)
        x2 = t1 > 0
        x3 = t1 * 0.5
        x4 = torch.where(x2, t1, x3)
        return x4
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
