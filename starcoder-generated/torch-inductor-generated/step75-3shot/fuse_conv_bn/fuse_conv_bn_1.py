
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, 3)
        self.conv_2 = torch.nn.Conv2d(1, 2, 3)
        self.conv_3 = torch.nn.Conv2d(2, 3, 3)
        self.bn_1 = torch.nn.BatchNorm2d(2)
        self.bn_2 = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        v1 = self.conv_1(x)
        v1 = self.bn_1(v1)
        v1 = self.bn_2(v1)
        v1 = self.conv_2(v1)
        v1 = self.conv_3(v1)
        return v1
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
