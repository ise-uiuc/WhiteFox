
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.rand(1)
        v3 = v1 - 0.3
        v4 = F.relu(v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
