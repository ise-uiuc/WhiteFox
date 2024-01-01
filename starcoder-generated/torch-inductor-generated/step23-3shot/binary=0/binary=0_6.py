
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, stride=3, padding=1)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v3 = v1 + other
        v4 = v3 + other
        v5 = v4 + other
        return v5
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
