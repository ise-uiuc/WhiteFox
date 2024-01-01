
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(100, 8, 1, stride=1, padding=0, groups=4)
    def forward(self, x5):
        t1 = self.conv_t(x5)
        t2 = t1 > 0
        t3 = t1 * 0.4201
        x7 = torch.where(t2, t1, t3)
        return x7
# Inputs to the model
x5 = torch.randn(32, 100, 12, 20)
