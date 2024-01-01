
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, min=0.1, max=0.1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 52, 52)
