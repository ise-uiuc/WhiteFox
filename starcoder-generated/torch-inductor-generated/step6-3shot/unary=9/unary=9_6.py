
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = torch.clamp(5*v2, 0, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
