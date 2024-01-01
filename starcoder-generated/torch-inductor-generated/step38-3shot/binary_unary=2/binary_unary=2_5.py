
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 - x2
        v3 = F.relu(v2)
        v4 = F.relu(x1 - self.conv(x2))
        v5 = v4.flatten(1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
x2 = torch.randn(1, 1, 4, 4)
