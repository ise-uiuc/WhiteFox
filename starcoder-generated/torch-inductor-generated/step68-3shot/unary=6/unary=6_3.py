
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 22, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.cat([v1, x1], dim=1)
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
