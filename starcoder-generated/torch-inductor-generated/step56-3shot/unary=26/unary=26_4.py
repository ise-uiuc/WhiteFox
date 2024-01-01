
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(11)
        self.conv_t = torch.nn.ConvTranspose2d(11, 11, 5, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(25)
    def forward(self, x8):
        j1 = self.bn1(x8)
        j2 = self.conv_t(j1)
        j3 = self.bn2(j2)
        j4 = j3 > 0
        j5 = j3 * 1.23
        j6 = torch.where(j4, j3, j5)
        return j6
# Inputs to the model
x8 = torch.randn(10, 11, 22, 12, device='cuda')
