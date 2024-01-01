
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t2 = torch.nn.ConvTranspose2d(1, 2, 1)
        self.conv_t3 = torch.nn.ConvTranspose2d(2, 3, 1)
        self.conv_t4 = torch.nn.ConvTranspose2d(3, 4, 1)
    def forward(self, x):
        x1 = self.conv_t2(x)
        x2 = self.conv_t3(x1)
        x3 = self.conv_t4(x2)
        x4 = x3 > 0
        x5 = x3 * 1.0
        x6 = torch.where(x4, x3, x5)
        return x6
# Inputs to the model
x = torch.randn(16, 1, 16, 16)
