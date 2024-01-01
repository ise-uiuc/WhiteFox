
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(54, 145, 2, stride=2, padding=0, bias=False)
    def forward(self, x0):
        r1 = self.conv_t(x0)
        r2 = r1 > 0
        r3 = r1 * -0.042842
        r4 = torch.where(r2, r1, r3)
        return r4
# Inputs to the model
x0 = torch.randn(1, 54, 67, 3)
