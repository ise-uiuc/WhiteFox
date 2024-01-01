
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.cat((v2, x2), dim=1)
        v4 = self.conv(v3)
        v5 = torch.relu(v4)
        v6 = torch.cat((v5, v2), dim=1)
        v7 = self.conv(v6)
        v8 = torch.add(v7, v5)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
