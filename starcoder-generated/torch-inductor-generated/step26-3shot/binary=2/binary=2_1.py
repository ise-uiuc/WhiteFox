
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.adaptiveavgpool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x0, x1):
        v0 = self.conv0(x0)
        v2 = self.conv1(x1)
        v1 = v0 - v2
        v4 = self.sigmoid(v1)
        v5 = self.adaptiveavgpool2d(v1)
        return v4
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
x1 = torch.randn(1, 3, 64, 64)
