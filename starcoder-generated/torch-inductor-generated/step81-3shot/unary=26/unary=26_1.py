
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(122, 184, 7, stride=3, padding=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(184, 210, 3, stride=1, padding=0)
    def forward(self, q):
        p1 = self.conv_t1(q)
        p2 = p1 > 0
        p3 = p1 * 0.33171
        p4 = torch.where(p2, p1, p3)
        p5 = self.conv_t2(p4)
        p6 = p5 > 0
        p7 = p5 * 0.20407
        p8 = torch.where(p6, p5, p7)
        return p8
# Inputs to the model
q = torch.randn(9, 122, 8, 12)
