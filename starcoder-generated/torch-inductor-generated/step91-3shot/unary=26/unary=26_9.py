
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(21, 13, 15, stride=5, padding=0, bias=False)
    def forward(self, x10):
        j1 = self.conv_t(x10)
        j2 = j1 > 0
        j3 = j1 * 0.278
        j4 = torch.where(j2, j1, j3)
        return j4
# Inputs to the model
x10 = torch.randn(4, 21, 13, 12)
