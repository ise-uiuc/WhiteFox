
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(3, 3, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(3)
    def forward(self, x3):
        v = self.bn(self.conv(x3))
        return v
# Inputs to the model
x3 = torch.randn(1, 3, 4, 4, 4)
