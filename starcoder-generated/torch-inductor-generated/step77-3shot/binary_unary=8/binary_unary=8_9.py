
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1.0
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
