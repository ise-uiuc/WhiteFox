
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 1, stride=1)
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 0
        t3 = t1 * -0.345
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x = torch.randn(1, 1, 10, 32)
