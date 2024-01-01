
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(337, 99, 3, stride=2, padding=3, output_padding=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(99, 3, 3, stride=1, padding=0, bias=True)
    def forward(self, x5):
        r1 = self.conv_t1(x5)
        r2 = r1 > 0.0
        r3 = r1 * -0.08
        r4 = torch.where(r2, r1, r3)
        r5 = self.conv_t2(r4)
        r6 = r5 > 0.0
        r7 = r5 * -0.41
        r8 = torch.where(r6, r5, r7)
        return r8
# Inputs to the model
x5 = torch.randn(43, 337, 19, 28)
