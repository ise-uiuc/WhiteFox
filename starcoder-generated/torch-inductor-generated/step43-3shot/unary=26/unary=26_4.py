
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 126, 7, stride=1, padding=3, bias=False)
    def forward(self, x14):
        l1 = self.conv_t(x14)
        l2 = l1 > 1e-05
        l3 = l1 * 0.0067
        l4 = torch.where(l2, l1, l3)
        return l4
# Inputs to the model
x14 = torch.randn(4, 256, 23, 34)
