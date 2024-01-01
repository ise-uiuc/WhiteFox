
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(137, 6, 7, stride=7, padding=0, bias=False)
    def forward(self, x4):
        x2 = self.conv_t(x4)
        n1 = x2 > 0
        n2 = x2 * -0.055843
        n3 = torch.where(n1, x2, n2)
        x5 = torch.nn.functional.sigmoid(n3)
        x6 = torch.add(x5, 2)
        return x6
# Inputs to the model
x4 = torch.randn(80, 137, 565)
