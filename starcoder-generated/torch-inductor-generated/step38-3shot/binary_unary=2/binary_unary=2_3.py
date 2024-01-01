
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 15
        v3 = F.relu(v2)
        v4 = v3 - self.conv(x1)
        v5 = F.relu(v4)
        v6 = self.conv(v1 - 5)
        v7 = v6 - v2
        v8 = self.conv(v7)
        v9 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
