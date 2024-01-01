
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(70, 70, 7, stride=1, padding=0, bias=False)
    def forward(self, x27):
        r1 = self.conv_t(x27)
        r2 = r1 > 0
        r3 = r1 * -247
        r4 = torch.where(r2, r1, r3)
        return r4
# Inputs to the model
x27 = torch.randn(2, 70, 82, 1)
