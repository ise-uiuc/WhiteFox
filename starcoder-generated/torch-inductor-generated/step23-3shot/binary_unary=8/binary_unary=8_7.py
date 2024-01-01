
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1.permute(0, 2, 3, 1)
        v4 = v2.permute(0, 2, 3, 1)
        v5 = torch.relu(v3 + v4)
        v6 = v5
        v7 = v6
        v8 = v7
        v9 = v8.permute(0, 3, 1, 2)
        v10 = v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
