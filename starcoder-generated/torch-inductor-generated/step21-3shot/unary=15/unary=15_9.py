
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 16, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.nn.functional.interpolate(v2, None, 2, 'nearest')
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 96, 96)
