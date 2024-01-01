
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 16, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = 2 * self.conv(x)
        v2 = 3 + v1
        v3 = v2 - 5
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 64, 64, 64)
