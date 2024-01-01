
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v7 = self.conv(x1)
        v8 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + x1 + x1
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
