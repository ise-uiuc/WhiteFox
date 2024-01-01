
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 1, 2)
        self.bn = torch.nn.BatchNorm3d(1)
    def forward(self, x):
        return self.bn(self.conv1(x))
# Inputs to the model
x = torch.randn(1, 1, 2, 2, 2)
