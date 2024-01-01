
class Model(torch.nn.Module):
    def __init__(self, param3):
        super(Model, self).__init__()
        self.conv4 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=0)
        self.conv_t5 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0, output_padding=0)
        self.conv_t6 = torch.nn.ConvTranspose2d(param3, 3, 1, stride=1, padding=0, output_padding=0)
        self.param0 = torch.nn.Parameter(0.5, requires_grad=False)
        self.param1 = torch.nn.Parameter(torch.tensor([param3]), requires_grad=False)
    def forward(self, x8):
        z1 = self.conv4(x8)
        z2 = self.conv_t5(z1)
        z3 = self.param0 *.91
        z4 = self.param0 *.61
        z5 = z3 - self.param1
        z6 = z3 > 0
        z7 = z5 - z4
        z8 = z3.where(z6, z7)
        z9 = self.param0 > 0
        z10 = z5 - z4
        z11 = z3.where(z9, z10)
        z12 = self.conv_t6(z11)
        z13 = self.param0 *.52
        z14 = self.param0 *.48
        z15 = z13 - self.param1
        z16 = z13 > 0
        w0 = z15 - z14
        z17 = z13.where(z16, w0)
        z18 = self.param0 > 0
        z19 = z15 - z14
        z20 = z13.where(z18, z19)
        return z2 - z17 + z20
param3 = 4
# Inputs to the model
x8 = torch.randn(9, 1, 10, 12)
