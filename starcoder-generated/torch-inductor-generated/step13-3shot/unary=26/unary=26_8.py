
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(72, 72, (3, 3), stride=1, padding=(1, 1), bias=False)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2 >= 1.0
        x4 = x2 * 5.398
        x5 = torch.where(x3, x1, x4)
        return x5
# Inputs to the model
x1 = torch.randn(10, 72, 24, 24)
