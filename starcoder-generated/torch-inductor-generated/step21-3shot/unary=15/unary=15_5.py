
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v0 = torch.nn.functional.interpolate(x1, None, 0.25, 'bilinear', True)
        v1 = self.conv1(v0)
        v2 = torch.nn.functional.interpolate(v1, None, 0.25, 'bilinear', True)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 257, 257)
