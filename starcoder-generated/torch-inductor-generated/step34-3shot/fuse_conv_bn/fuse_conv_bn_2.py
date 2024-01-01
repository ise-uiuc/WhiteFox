
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 5, 3)
        self.bn = torch.nn.BatchNorm1d(5)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x1):
        y = self.conv1(x1)
        z = self.bn(y)
        t = self.activation(z)
        return t
