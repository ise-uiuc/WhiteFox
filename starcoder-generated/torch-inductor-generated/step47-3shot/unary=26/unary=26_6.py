
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(134, 135, 7, stride=3, padding=0, bias=False)
    def forward(self, x3):
        j1 = self.conv_t(x3)
        j2 = j1 > 0
        j3 = j1 * 0.0044
        j4 = torch.where(j2, j1, j3)
        return torch.flatten(j4, 1)
# Inputs to the model
x3 = torch.randn(41, 134, 9, 77)
