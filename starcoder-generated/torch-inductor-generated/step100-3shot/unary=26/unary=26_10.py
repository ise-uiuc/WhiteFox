
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(22, 27, (2, 4), stride=1, padding=(0, 3), bias=False)
    def forward(self, x9):
        l1 = self.conv_t(x9)
        l2 = l1 > 0
        l3 = l1 * -0.0861
        l4 = torch.where(l2, l1, l3)
        return l4
# Inputs to the model
x9 = torch.randn(1, 22, 11, 18)
