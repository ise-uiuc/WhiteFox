
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm3d(1)
        self.conv1 = torch.nn.Conv3d(1, 1, 1)
    def forward(self, x1):
        a = self.bn1(x1)
        b = self.conv1(a)
        return b
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6, 6)
