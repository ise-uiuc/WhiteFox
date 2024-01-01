
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x27):
        z26 = x27
        w4 = self.conv(z26)
        w5 = self.bn(w4)
        w5 = self.bn(w5)
        z26 = w5
        w6 = self.conv(z26)
        w7 = self.bn(w6)
        w8 = self.bn(w7)
        return w8
# Inputs to the model
x27 = torch.randn(1, 3, 5, 5, 3)
