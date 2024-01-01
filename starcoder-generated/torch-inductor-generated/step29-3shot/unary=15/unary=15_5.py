
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 256, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.interpolate(v1, None, 2, 'nearest')
        v3 = torch.nn.functional.interpolate(v2, None, 4, 'nearest')
        v4 = torch.nn.functional.interpolate(v3, None, 8, 'nearest')
        v5 = torch.nn.functional.interpolate(v4, None, 16, 'nearest')
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 1, 1)
