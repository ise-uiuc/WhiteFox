
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.0
        v3 = F.relu(v2)
        v4 = torch.mean(v3, 1, True)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
