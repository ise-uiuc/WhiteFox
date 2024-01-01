
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, kernel_size=(3, 3, 3), padding=(2, 1, 3))
        self.bn = torch.nn.BatchNorm3d(3)

    def forward(self, x3):
        x3 = self.conv(x3)
        x4 = self.bn(x3)
        return x4
# Inputs to the model
x3 = torch.randn(1, 3, 9, 9, 9)
