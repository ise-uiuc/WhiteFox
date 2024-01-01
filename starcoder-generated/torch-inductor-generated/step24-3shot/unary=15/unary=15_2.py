
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.bn4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn5 = torch.nn.BatchNorm2d(8)
        self.bn6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn7 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn8 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.bn2(v1)
        v3 = self.bn3(v2)
        v4 = self.bn4(v3)
        v5 = self.bn5(v4)
        v6 = self.bn6(v5)
        v7 = self.bn7(v6)
        v8 = self.bn8(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 480, 640)
