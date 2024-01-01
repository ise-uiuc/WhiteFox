
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
        self.bn3 = torch.nn.BatchNorm2d(2, affine=False)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.bn1(x1)
        y2 = self.relu(self.bn2(y1))
        y3 = self.relu(self.bn3(self.relu(y2)))
        return y3
# Inputs to the model
x1 = torch.randn(3, 3, 4, 4)
