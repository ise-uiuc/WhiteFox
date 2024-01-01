
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 16, 5, stride=1, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2)
    def forward(self, x50):
        s1 = self.conv_t1(x50)
        x1 = s1 > 0
        x2 = s1 * -0.138
        x3 = torch.where(x1, s1, x2)
        x4 = self.conv_t2(x3)
        x5 = x4 > 0
        x6 = x4 * -0.149
        x7 = torch.where(x5, x4, x6)
        x8 = self.conv_t3(x7)
        x9 = x8 > 0
        x10 = x8 * -1.715
        s2 = torch.where(x9, x8, x10)
        return torch.nn.functional.adaptive_avg_pool2d(s2, (1, 1))
# Inputs to the model
x50 = torch.randn(1, 3, 30, 122)
