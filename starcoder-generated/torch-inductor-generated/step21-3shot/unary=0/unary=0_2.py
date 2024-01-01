
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2,2, 2, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 * 0.5123
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = self.bn(v1)
        v10 = v9 * 0.2938
        v11 = v10 * v8
        v12 = v11 + v6
        return v12
# Inputs to the model
x5 = torch.randn(1, 2, 16, 16)
