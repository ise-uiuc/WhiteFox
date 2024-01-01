
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.conv1(v2)
        v3 = self.bn1(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
