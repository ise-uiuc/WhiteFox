
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        t0 = torch.zeros_like(x1)
        v1 = self.conv1(t0)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        return v3 ** 0.5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
