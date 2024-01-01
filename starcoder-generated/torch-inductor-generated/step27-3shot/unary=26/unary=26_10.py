
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(36, 50, 1, stride=1)
        self.conv_t4 = torch.nn.ConvTranspose2d(50, 1, 2, stride=2)
        self.conv_t5 = torch.nn.ConvTranspose2d(1, 1, 1, stride=1)
        self.conv_t6 = torch.nn.ConvTranspose2d(1, 1, 4, stride=4)
        self.conv_t7 = torch.nn.ConvTranspose2d(1, 1, 8, stride=8)
    def forward(self, x):
        y = self.conv(x)
        z1 = self.conv_t4(y)
        z2 = self.conv_t5(z1)
        z3 = self.conv_t6(z2)
        z4 = self.conv_t7(z3)

        return z4
# Inputs to the model
x = torch.randn(1, 36, 32, 32)
