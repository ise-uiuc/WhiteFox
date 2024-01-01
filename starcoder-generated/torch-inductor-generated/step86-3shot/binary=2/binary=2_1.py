
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.add(self.conv(x1), 1)
        v2 = v1 - 3.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
