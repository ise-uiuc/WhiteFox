
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1)
    def forward(self, x1):
        v1 = x1 + 10
        v2 = F.relu(v1)
        v3 = self.conv(v2)
        v4 = v3 - 1
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
