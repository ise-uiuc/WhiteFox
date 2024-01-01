
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 2, 2)
        self.conv2 = torch.nn.Conv3d(2, 3, 2)
        self.bn1 = torch.nn.BatchNorm3d(2)
        self.bn2 = torch.nn.BatchNorm3d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        y = self.bn1(s)
        u = self.conv2(y)
        z = self.bn2(u)
        return z
# Inputs to the model
x1 = torch.randn(1, 1, 2, 4, 5)
