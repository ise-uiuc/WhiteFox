
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(32, 84, 4, stride=2, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(84, 168, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = t1 > 0
        t3 = t1 * -0.236
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * -0.293
        t8 = torch.where(t6, t5, t7)
        return t8
# Inputs to the model
x1 = torch.randn(16, 32, 26, 48)
