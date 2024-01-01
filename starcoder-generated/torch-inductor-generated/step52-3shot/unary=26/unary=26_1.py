
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3136, 3008, 10, stride=1, padding=0, bias=False)
    def forward(self, x7):
        t1 = self.conv_t(x7)
        t2 = t1 > 0
        t3 = t1 * 0.2570487
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x7 = torch.randn(65535, 3136, 24, 25)
