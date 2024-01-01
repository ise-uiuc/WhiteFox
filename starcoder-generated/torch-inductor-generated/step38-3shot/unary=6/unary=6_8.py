
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.swish = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout2d(p=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.bn(v5)
        v7 = self.swish(v6)
        v8 = self.dropout(v7)
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
