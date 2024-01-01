
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=12):
        v1 = self.conv(x1)
        v2 = other * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
