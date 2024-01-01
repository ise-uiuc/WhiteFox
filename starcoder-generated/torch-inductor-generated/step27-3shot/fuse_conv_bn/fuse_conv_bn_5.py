
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(16, 16, 3)
        torch.manual_seed(2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        torch.manual_seed(3)
        self.bn2 = torch.nn.BatchNorm2d(16)
        torch.manual_seed(4)
        self.bn3 = torch.nn.BatchNorm1d(16)
        torch.manual_seed(1)
    def forward(self, x0):
        v0 = self.bn1(x0)
        v0 = self.bn2(x0)
        v0 = F.relu(v0)
        v0 = self.bn2(v0)
        v0 = self.bn3(v0)
        v0 = F.relu(v0)
        v0 = self.bn2(v0)
        return v0
# Inputs to the model
x0 = torch.randn(1, 16, 3, 3)
