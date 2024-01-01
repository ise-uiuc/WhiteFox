
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(14, 22, (6, 8), padding=(4, 7), bias=False, stride=2)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0.5
        x3 = x1 * 10
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(12, 14, 10, 6)
