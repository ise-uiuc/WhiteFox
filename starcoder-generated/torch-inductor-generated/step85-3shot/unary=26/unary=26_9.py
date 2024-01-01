
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(22, 162, 3, padding=1, stride=1, bias=False)
    def forward(self, x2):
        l3 = self.conv_t(x2)
        l4 = l3 > 0
        l5 = l3 * -3.70
        l6 = torch.where(l4, l3, l5)
        return l6
# Input to the model
x2 = torch.randn(23, 22, 10, 10)
