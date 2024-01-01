
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, (1, 7), (1, 1), (0, 3))
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
