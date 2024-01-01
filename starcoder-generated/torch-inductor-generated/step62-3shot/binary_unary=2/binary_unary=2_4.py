
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 12, 1, stride=1, padding=0)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = v2 - 3
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
