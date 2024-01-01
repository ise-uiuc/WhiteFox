
class Model(torch.nn.Module):
    def __init__(self):
        super().init()
        self.conv = torch.nn.Conv2d(3, 15, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.3
        v3 = F.relu(v2)
        v4 = v3 - 0.2
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
