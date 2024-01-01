
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.max(x1, torch.randn(1, 1, 4, 1, 1))
        v2 = self.conv(v1)
        v3 = v2
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = torch.div(v4, 6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
