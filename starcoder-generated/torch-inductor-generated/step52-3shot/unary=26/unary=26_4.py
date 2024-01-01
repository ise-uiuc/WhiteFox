
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(25, 27, 4, padding=1, groups=29, bias=False)
    def forward(self, x0):
        n1 = self.conv_t(x0)
        n2 = n1 > 0
        n3 = n1 * 4.683
        n4 = torch.where(n2, n1, n3)
        return n4
# Inputs to the model
x0 = torch.randn(29, 25, 14, 10)
