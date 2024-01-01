
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 1, 2)
        self.bn = torch.nn.BatchNorm3d(1)
    def forward(self, x):
        x = self.conv1(x)
        y = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4, 4)
