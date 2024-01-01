
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 2, (6, 9), stride=(1, 2), padding=(4, 6), bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(2, 1, 3, stride=1, padding=1, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(5, 3, 5, stride=2, padding=2, bias=False)
    def forward(self, x2):
        y3 = self.conv_t1(x2)
        u3 = y3 > 0
        u2 = y3 * 0.35431211
        u1 = torch.where(u3, y3, u2)
        y2 = torch.neg(u1)
        y1 = self.conv_t2(y2)
        z2 = y1 > 0
        z3 = y1 * -0.28928739
        z1 = torch.where(z2, y1, z3)
        y4 = self.conv_t3(z1)
        v2 = y4 > 0
        v3 = (y4 * -0.00017121)
        v1 = torch.where(v2, y4, v3)
        return v1
# Inputs to the model
x2 = torch.randn(2, 3, 7, 7)
