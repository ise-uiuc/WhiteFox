
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.6
        v3 = F.relu(v2)
        v4 = v3 - 0.8
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
