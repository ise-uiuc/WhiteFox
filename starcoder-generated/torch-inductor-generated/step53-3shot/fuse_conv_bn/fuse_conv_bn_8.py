
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(3)
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x):
        v = self.conv(x)
        v0 = torch.add(v, 1)
        v1 = self.bn(v)
        v2 = self.conv1(v1)
        v3 = self.bn1(v2)
        v4 = self.conv1(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 3, 3, requires_grad=True)
