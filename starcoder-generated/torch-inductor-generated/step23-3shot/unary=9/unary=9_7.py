
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = 3 + v2
        v4 = v3.clamp(min=0, max=6)
        v5 = v4.div(6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
