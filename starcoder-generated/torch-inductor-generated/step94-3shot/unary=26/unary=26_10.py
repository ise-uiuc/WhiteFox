
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(3, 7, 3, stride=2, dilation=2, groups=4, padding=0, bias=True)
    def forward(self, x34):
        t1 = self.conv_t(x34)
        t2 = t1 > 0
        t3 = t1 * 0.0481903
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x34 = torch.randn(23, 3, 99)
