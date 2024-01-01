
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 3, 3)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 2, 1)
        self.conv_t3 = torch.nn.ConvTranspose2d(2, 9, 2, stride=2)
    def forward(self, v):
        f1 = self.conv_t1(v)
        f2 = f1 > 0
        f3 = f1 * -0.632
        f4 = torch.where(f2, f1, f3)
        f5 = self.conv_t2(f4)
        f6 = f5 > 0
        f7 = f5 * -0.530
        f8 = torch.where(f6, f5, f7)
        f9 = self.conv_t3(f8)
        f10 = f9 > 0
        f11 = f9 * -0.769
        f12 = torch.where(f10, f9, f11)
        return f12
# Inputs to the model
v = torch.randn(10, 3, 9, 28)
