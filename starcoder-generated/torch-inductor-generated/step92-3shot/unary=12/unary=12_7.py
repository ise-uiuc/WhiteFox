
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 18, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d(5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.adaptiveavgpool(v1)
        v4 = v2 * v3
        return v2 * v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
