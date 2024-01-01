
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v21 = v1 + v4
        v22 = v2 + v5
        v12 = v3 + v21 + v22
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
