
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3)
        self.bn = torch.nn.BatchNorm3d(3, momentum=0.4, eps=0.2)
    def forward(self, x1):
        self.conv(x1)
        t = self.bn(x1)
        return torch.add(t, t)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4, 4)
