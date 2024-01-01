
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 44, 3, stride=1, padding=2, bias=False)
    def forward(self, x1):
        j1 = self.conv_t(x1)
        j2 = j1 > 0
        j3 = j1 * 1.981
        j4 = torch.where(j2, j1, j3)
        return j4
# Inputs to the model
x1 = torch.randn(6, 2, 73, 89)
