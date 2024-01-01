
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(16, 7, 3)
        self.conv_t2 = torch.nn.ConvTranspose2d(7, 64, 11, stride=3)
        self.conv_t3 = torch.nn.ConvTranspose2d(64, 83, 7)
        self.conv_t4 = torch.nn.ConvTranspose2d(83, 163, 8, stride=3)
    def forward(self, v):
        c = self.conv_t1(v)
        d = c > 0
        e = c * -0.599
        f = torch.where(d, c, e)
        g = c + -0.633
        h = self.conv_t2(f)
        i = h > 0
        j = h * -0.21
        k = torch.where(i, h, j)
        l = f * 0.633
        m = l + k
        n = self.conv_t3(m)
        o = n > 0
        p = n * 0.818
        q = torch.where(o, n, p)
        r = c * -0.229
        s = r + v
        t = self.conv_t4(s)
        u = t > 0
        v = t * 0.831
        w = torch.where(u, t, v)
        return w
# Inputs to the model
v = torch.randn(1, 16, 10, 9)
