
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv3d(2, 2, 3)
        self.bn3 = torch.nn.BatchNorm3d(2)
        self.c1 = torch.nn.Conv3d(2, 3, 3)
        self.c2 = torch.nn.Conv3d(3, 4, 3)
        self.bn1 = torch.nn.BatchNorm3d(4)
        self.bn2 = torch.nn.BatchNorm3d(5)
    def forward(self, x3):
        x3 = self.conv3(x3)
        y1 = self.bn3(x3)
        y1 = self.c1(y1)
        y2 = self.c2(y1)
        y2 = self.bn1(y2)
        return self.bn2(y2)
# Inputs to the model
x3 = torch.randn(1, 2, 4, 4, 4)
