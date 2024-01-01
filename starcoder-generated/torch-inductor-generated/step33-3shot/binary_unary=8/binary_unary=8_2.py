
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v4 = v1 + v2
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v8 = v4 + v5 + v6
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
