
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(16, 8, (3, 3, 3))
        self.bn = torch.nn.BatchNorm3d(8)
    def forward(self, x):
        x = self.conv1(x)
        y = self.bn(x)
        return y
# Inputs to the model
x = torch.randn(1, 16, 4, 16, 16)
