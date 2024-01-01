
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 529, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(89, 49, 9, stride=1, padding=4, groups=1)
    def forward(self, w1):
        r1 = self.conv_t1(w1)
        r2 = self.conv_t2(r1)
        r3 = r2 > 0
        r4 = r2 * -0.42
        r5 = torch.where(r3, r2, r4)
        return r5
# Inputs to the model
w1 = torch.randn(5, 1, 55, 79)
