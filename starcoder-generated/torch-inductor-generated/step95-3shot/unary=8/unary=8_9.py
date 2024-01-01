
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 6, 5, stride=3, padding=2, groups=7)
        self.t1 = torch.randn(1, 7, 15, 49)
    def forward(self, x1):
        c1 = 0.6483789937019348
        y1 = self.t1 * c1
        y2 = torch.clamp(y1, min=-1)
        y3 = torch.clamp(y2, max=1)
        o1 = y3.sign()
        y4 = o1.float() * 2 - 1
        r1 = torch.clamp(y4, min=0)
        o2 = o1 == 0
        z1 = o2.float() * r1
        b1 = torch.clamp(z1, min=-1, max=1)
        v1 = self.conv_transpose(b1)
        v2 = v1 + 0.805849418683916
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=10)
        v5 = v1 * v4
        v6 = v5 / 8.9792871
        return v6
# Inputs to the model
x1 = torch.randn(1, 7, 15, 49)
