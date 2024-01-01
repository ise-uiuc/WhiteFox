
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 12, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(1)
        v3 = torch.nn.functional.relu6(v2)
        v4 = v3 / 5
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(7, 7, 192, 256)
