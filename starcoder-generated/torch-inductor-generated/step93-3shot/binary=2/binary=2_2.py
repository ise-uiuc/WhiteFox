
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=True)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 36.278687
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
