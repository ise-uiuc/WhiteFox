
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(23, 79, 4, stride=2, padding=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(79, 19, 1, stride=1, padding=0, bias=False)
    def forward(self, t8):
        a1 = self.conv_t1(t8)
        a2 = 0.091 * a1
        a3 = a2 > 0
        a4 = a2 * -0.255
        a5 = torch.where(a3, a2, a4)
        a6 = self.conv_t2(a5)
        a7 = a6 > 0
        a8 = a6 * -0.988
        a9 = torch.where(a7, a6, a8)
        return a9
# Inputs to the model
t8 = torch.randn(3, 23, 25, 44)
