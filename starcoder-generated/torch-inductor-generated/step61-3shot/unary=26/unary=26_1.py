
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose1d(1, 1, 1)
    def forward(self, x18):
        f1 = self.conv_t1(x18)
        f2 = f1 > 0
        f3 = f1 * 0.273
        f4 = torch.where(f2, f1, f3)
        return f4
# Inputs to the model
x18 = torch.randn(7, 1, 46)
