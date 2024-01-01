
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(999, 870, 2, 1, 0, 1, 1)
    def forward(self, x2):
        r1 = self.conv_t(x2)
        r2 = r1 > 0
        r3 = r1 * 0.92
        r4 = torch.where(r2, r1, r3)
        return r4
# Inputs to the model
x2 = torch.randn(1, 999, 4, 7)
