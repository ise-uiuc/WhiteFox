
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = torch.add(v1, 1, v2)
        v4 = self.conv(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
