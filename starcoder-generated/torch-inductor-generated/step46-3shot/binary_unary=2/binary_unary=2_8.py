
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.ones_like(v1) + 0.1
        v3 = v2 - v1
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
