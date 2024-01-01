
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 3, 2)
        torch.manual_seed(0)
        self.bn1= torch.nn.BatchNorm2d(3)
        torch.manual_seed(0)
        self.bn2 = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.bn1(v1)
        v1 = self.bn2(v1)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 24, 24)
