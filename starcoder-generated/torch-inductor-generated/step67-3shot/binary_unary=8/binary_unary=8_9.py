
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v4 = self.conv(v3)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
